#include <chrono>
#include <iostream>
#include "decision_tree.h"

struct RandomForest {
  std::vector<DecisionTree *> trees;
  int max_depth_tree;
  int number_of_trees;
  DataSet dataset;
  RandomForest(DataSet &d_s, int m_d, int n_t)
      : dataset(d_s), max_depth_tree(m_d), number_of_trees(n_t){};
  void BuildForest();

  void Score(int num_of_queries,
             std::vector<std::vector<double>> queries_sample,
             std::vector<int> queries_target);
};

void RandomForest::BuildForest() {
  std::vector<int> attributes_indexes;
  for (int i = 0; i < dataset.total_features; i++) {
    attributes_indexes.push_back(i);
  }
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, dataset.total_features - 1);
  for (int i = 0; i < number_of_trees; i++) {
    std::vector<int> masked_attrs(dataset.total_features, 1);
    std::shuffle(attributes_indexes.begin(), attributes_indexes.end(), gen);
    for (int j = 0; j < sqrt(dataset.total_features); j++) {
      // std::cout << attributes_indexes[j] << " ";
      masked_attrs[attributes_indexes[j]] = 0;
    }
    // std::cout << std::endl;
    // std::cout << "\nFEATURES" << std::endl;
    // for (int j = 0; j < dataset.total_features; j++) {
    //   std::cout << masked_attrs[j] << " ";
    // }
    // std::cout << std::endl;
    dataset.masked_attributes = masked_attrs;
    DecisionTree *d_t = new DecisionTree();
    d_t->id = i;
    d_t->max_depth = max_depth_tree;
    d_t->rootNode = d_t->Build(dataset, 0);
    // std::cout << "TREE " << i << std::endl;
    // PrintTree(d_t->rootNode);
    trees.push_back(d_t);
  }
}

void RandomForest::Score(int num_of_queries,
                         std::vector<std::vector<double>> queries_sample,
                         std::vector<int> queries_target) {
  std::vector<std::unordered_map<int, int>> predictions;
  for (int i = 0; i < num_of_queries; i++) {
    std::unordered_map<int, int> p;
    // std::cout << "****Query " << i << "****" << std::endl;
    for (auto it : trees) {
      // std::cout << "Tree " << it->id << std::endl;
      int predicted_class = Predict(queries_sample[i], it->rootNode);
      // predictions[i][predicted_class] += 1;
      // std::cout << "predicted class: " << predicted_class
      //           << " target class: " << queries_target[i] << std::endl;
      p[predicted_class] += 1;
    }
    predictions.push_back(p);
  }

  double assertions = 0.0;
  // std::cout << "This is my array with predictions" << std::endl;
  for (int i = 0; i < predictions.size(); i++) {
    int aux_max = std::numeric_limits<int>::min();
    int max_class;
    for (auto it : predictions[i]) {
      // std::cout << it.first << " : " << it.second << std::endl;
      if (aux_max < it.second) {
        aux_max = it.second;
        max_class = it.first;
      }
    }
    if (max_class == queries_target[i]) {
      assertions += 1.0;
    }
    // std::cout << "mayority: " << max_class << " target " << queries_target[i]
    //           << std::endl;
    // std::cout << std::endl;
  }

  // std::cout << "Score" << std::endl;
  std::cout << assertions / num_of_queries << std::endl;

  // for (auto it : forest.trees) {
  //   for (int i = 0; i < num_of_queries; i ++) {
  //     clss = Predict(q, it->rootNode);
  //     std::cout << "class on predict return: " << clss << std::endl;
  //     predictions[clss] += 1;
  //   }
  // }
  // std::cout << "\nTotal predictions" << std::endl;
  // for (auto it : predictions) {
  //   std::cout << it.first << " " << ((double)it.second / num_trees)
  //             << std::endl;
  // }
  // std::cout << "Result: " << (double)predictions[target_q] / num_trees
  //           << std::endl;
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

  int total_features = data[0].size();

  DataSet dataset =
      DataSet(data, target_values, target_attributes, total_features);

  RandomForest forest = RandomForest(dataset, max_depth, num_trees);

  std::chrono::high_resolution_clock::time_point start =
      std::chrono::high_resolution_clock::now();
  forest.BuildForest();
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

  forest.Score(num_of_queries, queries_sample, queries_target);
  return 0;
}