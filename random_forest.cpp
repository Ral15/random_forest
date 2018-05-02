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
    d_t->max_depth = max_depth_tree;
    d_t->rootNode = d_t->Build(dataset, 0);
    std::cout << "TREE " << i << std::endl;
    PrintTree(d_t->rootNode);
    trees.push_back(d_t);
  }
}

int main() {
  int num_data, len_data, max_depth, num_trees;
  std::cin >> num_trees;
  std::cin >> max_depth;
  std::cin >> num_data;
  std::cin >> len_data;
  std::vector<std::vector<double>> data;
  // read dataset
  double x;
  for (int i = 0; i < num_data; i++) {
    std::vector<double> aux;
    for (int j = 0; j < len_data; j++) {
      std::cin >> x;
      aux.push_back(x);
    }
    data.push_back(aux);
  }
  // read target labels
  int y;
  std::vector<int> target_values;
  std::set<int> target_attributes;
  for (int i = 0; i < num_data; i++) {
    std::cin >> y;
    target_values.push_back(y);
    target_attributes.insert(y);
  }

  int total_features = data[0].size();

  DataSet dataset =
      DataSet(data, target_values, target_attributes, total_features);

  RandomForest forest = RandomForest(dataset, max_depth, num_trees);

  forest.BuildForest();

  std::vector<double> q = {842, 0,  2.2, 0,    1, 0, 7,  0.6, 188, 2,
                           2,   20, 756, 2549, 9, 7, 19, 0,   0,   1};
  int target_q = 1;
  std::cout << std::endl;
  std::cout << "Predictions" << std::endl;
  std::unordered_map<int, int> predictions;
  int clss;
  double prob;
  for (auto it : forest.trees) {
    clss = Predict(q, it->rootNode);
    std::cout << "class on predict return: " << clss << std::endl;
    predictions[clss] += 1;
  }
  std::cout << "\nTotal predictions" << std::endl;
  for (auto it : predictions) {
    std::cout << it.first << " " << ((double)it.second / num_trees)
              << std::endl;
  }
  std::cout << "Result: " << (double)predictions[target_q] / num_trees
            << std::endl;
  return 0;
}