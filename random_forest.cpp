#include <chrono>
#include <iostream>
#include <thread>
#include "decision_tree.h"

int NUMBER_OF_THREADS = std::thread::hardware_concurrency();

struct RandomForest {
  DataSet dataset_;
  int max_depth_tree_;
  int number_of_trees_;
  std::vector<DecisionTree *> trees_;
  RandomForest(DataSet &d_s, int m_d, int n_t)
      : dataset_(d_s),
        max_depth_tree_(m_d),
        number_of_trees_(n_t),
        trees_(n_t, nullptr) {
    // BuildForest(0, number_of_trees);
    std::vector<std::thread> threads;
    if (number_of_trees_ <= NUMBER_OF_THREADS) {
      for (int i = 0; i < number_of_trees_; i++) {
        // std::cout << "less than" << std::endl;
        threads.push_back(std::thread(&RandomForest::BuildForest, this, i));
      }
    } else {
      int batch_size =
          (number_of_trees_ + NUMBER_OF_THREADS - 1) / NUMBER_OF_THREADS;
      // std::cout << batch_size << std::endl;
      for (int i = 0; i < NUMBER_OF_THREADS; i++) {
        int start = i * batch_size;
        int end = std::min(start + batch_size, number_of_trees_);
        // std::cout << start << " " << end << std::endl;
        threads.push_back(
            std::thread(&RandomForest::BuildBatchForest, this, start, end));
      }
    }
    for (auto &th : threads) th.join();
  };
  void BuildBatchForest(int start, int end);
  void BuildForest(int id);

  void Score(int num_of_queries,
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
  std::vector<int> attributes_indexes =
      CreateAttributeIdxs(dataset_.num_of_features_);
  // for (int i = 0; i < end; i++) {
  dataset_.masked_attributes_ = SelectFeaturesRand(dataset_.num_of_features_);

  std::vector<int> sample_idxs;
  for (int i = 0; i < static_cast<int>(dataset_.data_.size()); i++) {
    sample_idxs.push_back(i);
  }

  // std::chrono::high_resolution_clock::time_point start =
  //     std::chrono::high_resolution_clock::now();

  DecisionTree *d_t =
      new DecisionTree(id, max_depth_tree_, dataset_, sample_idxs);

  // std::chrono::high_resolution_clock::time_point end =
  //     std::chrono::high_resolution_clock::now();
  // auto duration =
  //     std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
  //         .count();
  // std::cout << "Time building tree " << id << " : " << duration / 1000.0
  //           << std::endl;
  trees_[id] = d_t;
}

void RandomForest::BuildBatchForest(int start, int end) {
  // std::cout << start << " " << end << std::endl;
  std::vector<int> attributes_indexes =
      CreateAttributeIdxs(dataset_.num_of_features_);
  for (int i = start; i < end; i++) {
    dataset_.masked_attributes_ = SelectFeaturesRand(dataset_.num_of_features_);

    std::vector<int> sample_idxs;
    for (int j = 0; j < static_cast<int>(dataset_.data_.size()); j++) {
      sample_idxs.push_back(j);
    }

    // std::chrono::high_resolution_clock::time_point start =
    //     std::chrono::high_resolution_clock::now();

    DecisionTree *d_t =
        new DecisionTree(i, max_depth_tree_, dataset_, sample_idxs);

    // std::chrono::high_resolution_clock::time_point end =
    //     std::chrono::high_resolution_clock::now();
    // auto duration =
    //     std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
    //         .count();
    // std::cout << "Time building tree " << i << " : " << duration / 1000.0
    //           << std::endl;
    trees_[i] = d_t;
  }
}

void RandomForest::Score(int num_of_queries,
                         const std::vector<std::vector<double>> &queries_sample,
                         const std::vector<int> &queries_target) {
  std::vector<std::unordered_map<int, int>> predictions;
  for (int i = 0; i < num_of_queries; i++) {
    std::unordered_map<int, int> p;
    for (auto &it : trees_) {
      int predicted_class = Predict(queries_sample[i], it->rootNode_);
      p[predicted_class] += 1;
    }
    predictions.push_back(std::move(p));
  }
  double assertions = 0.0;
  for (int i = 0; i < static_cast<int>(predictions.size()); i++) {
    int aux_max = std::numeric_limits<int>::min();
    int max_class;
    for (auto &it : predictions[i]) {
      if (aux_max < it.second) {
        aux_max = it.second;
        max_class = it.first;
      }
    }
    if (max_class == queries_target[i]) {
      assertions += 1.0;
    }
  }

  std::cout << "Score => " << assertions / num_of_queries << std::endl;
}

int main() {
  int num_data, len_data, max_depth, num_trees;
  std::cin >> num_trees;
  std::cin >> max_depth;
  std::cin >> num_data;
  std::cin >> len_data;
  // read dataset
  std::vector<std::vector<double>> data = ReadSample(num_data, len_data);
  // read target labels
  std::vector<int> target_values;
  std::set<int> target_attributes;
  std::tie(target_values, target_attributes) = ReadTargetValues(num_data);

  int num_of_features = data[0].size();

  DataSet dataset =
      DataSet(data, target_values, target_attributes, num_of_features);

  std::chrono::high_resolution_clock::time_point start =
      std::chrono::high_resolution_clock::now();
  RandomForest forest = RandomForest(dataset, max_depth, num_trees);
  std::chrono::high_resolution_clock::time_point end =
      std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  std::cout << "Time building forest : " << duration / 1000.0 << std::endl;

  int num_of_queries;
  std::cin >> num_of_queries;

  std::vector<std::vector<double>> queries_sample =
      ReadSample(num_of_queries, len_data);

  std::vector<int> queries_target;
  std::tie(queries_target, std::ignore) = ReadTargetValues(num_of_queries);

  // std::chrono::high_resolution_clock::time_point start =
  //     std::chrono::high_resolution_clock::now();
  forest.Score(num_of_queries, queries_sample, queries_target);
  // std::chrono::high_resolution_clock::time_point end =
  //     std::chrono::high_resolution_clock::now();
  // auto duration =
  //     std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
  //         .count();
  // std::cout << "Time for predicting : " << duration / 1000.0 << std::endl;

  // forest.Score(num_of_queries, queries_sample, queries_target);
  return 0;
}