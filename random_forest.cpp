#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>
#include "decision_tree.h"

int NUMBER_OF_THREADS = std::thread::hardware_concurrency();
std::mutex mylock;
// const int seed = 1241230;

struct RandomForest {
  DataSet dataset_;
  int max_depth_tree_;
  int number_of_trees_;
  std::vector<DecisionTree *> trees_;
  std::vector<std::vector<std::unordered_map<int, int>>> predictions_;

  RandomForest(const RandomForest &f) = delete;
  RandomForest &operator=(const RandomForest &f) = delete;

  RandomForest(DataSet &d_s, int m_d, int n_t, int n_q)
      : dataset_(d_s),
        max_depth_tree_(m_d),
        number_of_trees_(n_t),
        trees_(n_t, nullptr),
        predictions_(n_q, std::vector<std::unordered_map<int, int>>(n_t)) {
    std::vector<std::thread> threads;
    // if (number_of_trees_ <= NUMBER_OF_THREADS) {
    //   for (int i = 0; i < number_of_trees_; i++) {
    //     threads.push_back(std::thread(&RandomForest::BuildForest, this, i));
    //   }
    // }
    if (number_of_trees_ == 1) {
      this->BuildForest(0);
    } else {
      int batch_size =
          (number_of_trees_ + NUMBER_OF_THREADS - 1) / NUMBER_OF_THREADS;
      for (int i = 0; i < NUMBER_OF_THREADS; i++) {
        // std::cout << "?" << std::endl;
        int start = i * batch_size;
        int end = std::min(start + batch_size, number_of_trees_);
        threads.push_back(
            std::thread(&RandomForest::BuildBatchForest, this, start, end));
      }
    }
    for (auto &th : threads) th.join();
  };
  ~RandomForest() { trees_.clear(); };
  void BuildBatchForest(int start, int end);
  void BuildForest(int id);

  std::vector<int> Predict(
      const int num_of_queries,
      const std::vector<std::vector<double>> &queries_sample);

  void PredictByBatch(const int start, const int end,
                      const std::vector<double> &query, const int q_p);

  double Score(const int num_of_queries,
               const std::vector<std::vector<double>> &queries_sample,
               const std::vector<int> &queries_target);
};

std::vector<int> CreateAttributeIdxs(int number_features) {
  std::vector<int> attributes_indexes;
  for (int i = 0; i < number_features; i++) {
    attributes_indexes.push_back(i);
  }
  return attributes_indexes;
}

std::vector<int> SelectFeaturesRand(int number_features) {
  std::vector<int> attributes_indexes = CreateAttributeIdxs(number_features);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, number_features - 1);
  std::vector<int> masked_attrs(number_features, 1);
  std::shuffle(attributes_indexes.begin(), attributes_indexes.end(), gen);
  for (int i = 0; i < sqrt(number_features); i++) {
    masked_attrs[attributes_indexes[i]] = 0;
  }
  return masked_attrs;
}

void RandomForest::BuildForest(int id) {
  // std::cout << id << std::endl;
  std::vector<int> attributes_indexes =
      CreateAttributeIdxs(dataset_.num_of_features_);
  auto masked_attributes = SelectFeaturesRand(dataset_.num_of_features_);

  std::vector<int> sample_idxs;
  for (int i = 0; i < static_cast<int>(dataset_.data_.size()); i++) {
    sample_idxs.push_back(i);
  }

  DecisionTree *d_t = new DecisionTree(id, max_depth_tree_, dataset_,
                                       sample_idxs, masked_attributes);

  // std::cout << "?" << std::endl;

  trees_[id] = d_t;
}

void RandomForest::BuildBatchForest(int start, int end) {
  try {
    std::vector<int> attributes_indexes =
        CreateAttributeIdxs(dataset_.num_of_features_);
    for (int i = start; i < end; i++) {
      auto masked_attributes = SelectFeaturesRand(dataset_.num_of_features_);

      std::vector<int> sample_idxs;
      for (int j = 0; j < static_cast<int>(dataset_.data_.size()); j++) {
        sample_idxs.push_back(j);
      }

      DecisionTree *d_t = new DecisionTree(i, max_depth_tree_, dataset_,
                                           sample_idxs, masked_attributes);

      // mylock.lock();
      trees_[i] = d_t;
      // mylock.unlock();
    }

  } catch (std::exception e) {
    std::cout << "Se chalequio esta madre" << std::endl;
    std::cerr << e.what() << std::endl;
  }
}

int GetBatchSize(const int tasks) {
  return (tasks + NUMBER_OF_THREADS - 1) / NUMBER_OF_THREADS;
}

std::vector<int> RandomForest::Predict(
    const int num_of_queries,
    const std::vector<std::vector<double>> &queries_sample) {
  for (int q_p = 0; q_p < num_of_queries; q_p++) {
    // don't parallelize
    if (number_of_trees_ == 1) {
      for (auto &it : trees_) {
        int predicted_class =
            it->PredictTree(queries_sample[q_p], it->rootNode_);
        predictions_[q_p][0][predicted_class] += 1;
      }
    } else {
      // predict in parallel
      std::vector<std::thread> threads;
      int batch_size = GetBatchSize(number_of_trees_);
      for (int i = 0; i < NUMBER_OF_THREADS; i++) {
        int start = i * batch_size;
        int end = std::min(start + batch_size, number_of_trees_);
        threads.push_back(std::thread(&RandomForest::PredictByBatch, this,
                                      start, end, queries_sample[q_p], q_p));
      }
      for (auto &th : threads) th.join();
    }
  }
  std::vector<int> class_votes;
  for (auto &q_p : predictions_) {
    int best_class;
    int total_p = -1;
    for (auto &tree : q_p) {
      for (auto &prediction : tree) {
        if (total_p < prediction.second) {  // what if there is a tie?
          best_class = prediction.first;
        }
      }
    }
    class_votes.push_back(best_class);
  }
  return class_votes;
}

void RandomForest::PredictByBatch(const int start, const int end,
                                  const std::vector<double> &query,
                                  const int q_p) {
  // std::cout << "query point: " << q_p << "start & end: " << start << " " <<
  // end
  //           << std::endl;
  // trees
  // std::unordered_map<int, int> p;
  for (int i = start; i < end; i++) {
    int predicted_class =
        DecisionTree::PredictTree(query, trees_[i]->rootNode_);
    // std::cout << i << " " << predicted_class << std::endl;
    // p[predicted_class] += 1;
    predictions_[q_p][i][predicted_class] += 1;
  }

  // predictions_[q_p] +=

  // std::unordered_map<int, int> p;
  // for (int i = start; i < end; i++) {
  //   for (auto &it : trees_) {
  //     // std::cout << "this is i: " << i << std::endl;
  //     int predicted_class = DecisionTree::PredictTree(query, it->rootNode_);
  //     p[predicted_class] += 1;
  //   }
  //   // std::cout << start << " " << end << std::endl;
  //   // std::cout <<
  // }
  // predictions_[q_p] = std::move(p);
}

double RandomForest::Score(
    int num_of_queries, const std::vector<std::vector<double>> &queries_sample,
    const std::vector<int> &queries_target) {
  // std::vector<std::unordered_map<int, int>> predictions;
  // for (int i = 0; i < num_of_queries; i++) {
  //   std::unordered_map<int, int> p;
  //   for (auto &it : trees_) {
  //     int predicted_class =
  //         DecisionTree::Predict(queries_sample[i], it->rootNode_);
  //     p[predicted_class] += 1;
  //   }
  //   predictions.push_back(std::move(p));
  // }
  // std::ignore = this->Predict(num_of_queries, queries_sample);
  auto predictions = this->Predict(num_of_queries, queries_sample);

  double assertions = 0.0;
  for (int i = 0; i < static_cast<int>(predictions.size()); i++) {
    if (predictions[i] == queries_target[i]) {
      assertions += 1.0;
    }
  }

  // for (int i = 0; i < num_of_queries; i++) {
  //   int max_freq = std::numeric_limits<int>::min();
  //   int max_class;
  //   for (auto &tree : predictions_[i]) {
  //     for (auto &prediction : tree) {
  //       std::cout << prediction.first << " " << prediction.second;
  //       if (max_freq < prediction.second) {
  //         max_class = prediction.first;
  //         max_freq = prediction.second;
  //       }
  //     }
  //     std::cout << std::endl;
  //   }
  //   // std::cout << max_class << " " << queries_target[i] << " id: " << i
  //   //           << std::endl;
  //   if (max_class == queries_target[i]) {
  //     assertions += 1.0;
  //   }
  // }

  // std::cout << assertions / num_of_queries << std::endl;
  return assertions / num_of_queries;

  // for (auto &tree: predictions_) {
  //   for (int i = 0; )
  // }

  // for (auto it: predictions_)

  // double assertions = 0.0;
  // for (int i = 0; i < static_cast<int>(predictions_.size()); i++) {
  //   int aux_max = std::numeric_limits<int>::min();
  //   int max_class;
  //   for (auto &it : predictions_[i]) {
  //     // std::cout << it.first << " " << it.second << std::endl;
  //     if (aux_max < it.second) {
  //       aux_max = it.second;
  //       max_class = it.first;
  //     }
  //   }
  //   if (max_class == queries_target[i]) {
  //     assertions += 1.0;
  //   }
  // }

  // // std::cout << "Score => " << assertions / num_of_queries << std::endl;
  // return assertions / num_of_queries;
}

int main() {
  int num_data, len_data, max_depth, num_trees;
  std::cin >> num_trees;
  std::cin >> max_depth;
  std::cin >> num_data;
  std::cin >> len_data;

  std::vector<std::vector<double>> data = ReadSample(num_data, len_data);

  std::vector<int> target_values;
  std::set<int> target_attributes;
  std::tie(target_values, target_attributes) = ReadTargetValues(num_data);

  int num_of_features = data[0].size();

  DataSet dataset =
      DataSet(data, target_values, target_attributes, num_of_features);

  int num_of_queries;
  std::cin >> num_of_queries;

  std::vector<std::vector<double>> queries_sample =
      ReadSample(num_of_queries, len_data);

  std::vector<int> queries_target;

  std::tie(queries_target, std::ignore) = ReadTargetValues(num_of_queries);

  std::cout << "Number of trees: " << num_trees << std::endl;
  std::cout << "Max Depth Trees: " << max_depth << std::endl;

  double scores, avg_p_time, avg_b_time;
  scores = avg_b_time = avg_p_time = 0.0;
  for (int i = 1; i <= 10; i++) {
    // std::cout << i << std::endl;
    std::chrono::high_resolution_clock::time_point start =
        std::chrono::high_resolution_clock::now();
    RandomForest forest(dataset, max_depth, i, num_of_queries);
    std::chrono::high_resolution_clock::time_point end =
        std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    avg_b_time += duration / 1000.0;
    std::cout << "(" << i << ", " << duration / 1000.0 << ")" << std::endl;

    std::chrono::high_resolution_clock::time_point start_predict =
        std::chrono::high_resolution_clock::now();

    // std::cout << "this is the prediction: "
    //           << forest.Predict(num_of_queries, queries_sample) << std::endl;

    double score = forest.Score(num_of_queries, queries_sample, queries_target);
    // std::cout << "Score: " << score << std::endl; scores +=
    // score;
    std::chrono::high_resolution_clock::time_point end_predict =
        std::chrono::high_resolution_clock::now();
    auto duration_p = std::chrono::duration_cast<std::chrono::milliseconds>(
                          end_predict - start_predict)
                          .count();
    avg_p_time += duration_p / 1000.0;
    std::cout << "(" << i << ", " << duration_p / 1000.0 << ")" << std::endl;
    std::cout << score << std::endl;
    // std::cout << "Time predicting: " << duration_p / 1000.0 << std::endl;

    // for (int i = 0; i < static_cast<int>(forest.trees_.size()); i++) {
    //   // std::cout << &forest.trees_[i] << std::endl;
    //   free(forest.trees_[i]);
    //   forest.trees_[i] = nullptr;
    // }
  }

  // std::cout << "\n*****\nAvg score: " << scores / 10.0 << std::endl;
  // std::cout << "Avg building time: " << avg_b_time / 10.0 << std::endl;
  // std::cout << "Avg predict time: " << avg_p_time / 10.0 << std::endl;
  return 0;
}