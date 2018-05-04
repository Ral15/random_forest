#include <math.h>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "dataset.h"
#include "node.h"

struct DecisionTree {
  int id_;
  int max_depth_;
  Node *rootNode_;
  Node *Build(const DataSet &d, int curr_depth, const std::vector<int> &sample_idxs);
  DecisionTree(int m_d, int id, const DataSet &d, const std::vector<int> &sample_idxs): 
  id_(id), max_depth_(m_d), rootNode_(Build(d, 0, sample_idxs)) {};
};

void PrintTabs(int tabs) {
  for (int i = 0; i < tabs; i++) std::cout << "|";
}

// void PrintTree(Node *curr_node) {
//   if (curr_node == nullptr) {
//     return;
//   }
//   if (!curr_node->is_leaf) {
//     PrintTabs(curr_node->curr_depth);
//     std::cout << "X[" << curr_node->feature
//               << "] <= " << curr_node->splitted_value << std::endl;
//     PrintTabs(curr_node->curr_depth);
//     std::cout << "gini = " << curr_node->gini_index << std::endl;
//     PrintTabs(curr_node->curr_depth);
//     std::cout << "samples = " << curr_node->sample_size << std::endl;
//     PrintTabs(curr_node->curr_depth);
//     std::cout << "values => ";
//     for (auto it : curr_node->frequency) {
//       std::cout << it.second << " ";
//     }
//     std::cout << std::endl;
//     PrintTabs(curr_node->curr_depth);
//     std::cout << "*Left*\n";
//     PrintTree(curr_node->left_child);
//     PrintTabs(curr_node->curr_depth);
//     std::cout << "*Right*\n";
//     PrintTree(curr_node->right_child);
//   } else {
//     PrintTabs(curr_node->curr_depth);
//     std::cout << "gini = " << curr_node->gini_index << std::endl;
//     PrintTabs(curr_node->curr_depth);
//     std::cout << "samples = " << curr_node->sample_size << std::endl;
//     PrintTabs(curr_node->curr_depth);
//     std::cout << "values => ";
//     for (auto it : curr_node->frequency) {
//       std::cout << it.second << " ";
//     }
//     std::cout << std::endl;
//     PrintTabs(curr_node->curr_depth);
//     curr_node->Classify();
//     // std::cout << "ENDDDDD" << std::endl;
//     return;
//   }
// }

/*
  This is in case that the values for the class are different
*/
std::set<double> GetFeatureAttributes(
    const std::vector<std::vector<double>> &sample, int feature_index) {
  std::set<double> feature_attributes;
  for (int i = 0; i < static_cast<int>(sample.size()); i++) {
    feature_attributes.insert(sample[i][feature_index]);
  }
  return feature_attributes;
}

std::tuple<std::vector<int>, std::vector<int>> SplitSample(const DataSet &sample,
                                         const double split_value,
                                         const int idx_of_attribute,
                                         const std::vector<int> &sample_idxs) {
  std::vector<int> left_idxs;
  std::vector<int> right_idxs;
  for (int i = 0; i < static_cast<int>(sample_idxs.size()); i ++) {
    if (sample.data_[sample_idxs[i]][idx_of_attribute] <= split_value) {
      left_idxs.push_back(sample_idxs[i]);
    } else {
      right_idxs.push_back(sample_idxs[i]);
    }
  }
  return {left_idxs, right_idxs};
}

std::unordered_map<int, int> GetClassFrequency(const DataSet &sample,
                                               const std::set<int> &labels, 
                                               const std::vector<int> &sample_idxs) {
  std::unordered_map<int, int> frequency;
  for (auto &it : labels) {
    frequency[it];
  }
  for (int i = 0; i < static_cast<int>(sample_idxs.size()); i++) {
    frequency[sample.target_values_[sample_idxs[i]]] += 1;
  }
  return frequency;
}

double GiniIndex(const std::unordered_map<int, int> &frequency,
                 const int sample_size) {
  double gini_index = 0.0;
  for (auto &it : frequency) {
    gini_index +=
        (double)it.second / sample_size * (1 - (double)it.second / sample_size);
  }
  return gini_index;
}

double GiniSplit(const DataSet &sample, const double feature_attribute,
                 const int attribute_index, 
                 const std::vector<int> &sample_idxs) {
  std::unordered_map<int, int> left_freq;
  std::unordered_map<int, int> right_freq;
  int l, r;
  l = r = 0;
  for (int i = 0; i < static_cast<int>(sample_idxs.size()); i++) {
    if (sample.data_[sample_idxs[i]][attribute_index] <= feature_attribute) {
      left_freq[sample.target_values_[sample_idxs[i]]] += 1;
      l++;
    } else {
      right_freq[sample.target_values_[sample_idxs[i]]] += 1;
      r++;
    }
  }

  double left_gini =
      (((double)l / sample_idxs.size()) * GiniIndex(left_freq, l));
  double right_gini =
      (((double)r / sample_idxs.size()) * GiniIndex(right_freq, r));
  return left_gini + right_gini;
}

std::tuple<int, double, double> BestGini(
    const DataSet &sample, 
    const std::vector<int> &sample_idxs) {
  double best_gini = std::numeric_limits<double>::max();
  double best_attr_indx = std::numeric_limits<double>::min();
  double attr_value_split = std::numeric_limits<double>::min();
  for (int i = 0; i < sample.num_of_features_; i++) {
    if (sample.masked_attributes_[i]) {
      std::set<double> feature_attributes =
          GetFeatureAttributes(sample.data_, i);
      for (auto &attr : feature_attributes) {  // compute the gini index
        double new_gini = GiniSplit(sample, attr, i, sample_idxs);
        if (new_gini < best_gini) {  // update values
          best_gini = new_gini;
          best_attr_indx = i;
          attr_value_split = attr;
        }
      }
    }
  }
  std::tuple<int, double, double> gini_split =
      std::make_tuple(best_attr_indx, best_gini, attr_value_split);

  return gini_split;
}

int ShouldStop(const std::unordered_map<int, int> &frequency) {
  int freq_left = 1;
  for (auto &it : frequency) {
    if (it.second > 0) freq_left = 0;
  }
  return freq_left;
}



Node *DecisionTree::Build(const DataSet &sample, int curr_depth,
                          const std::vector<int> &sample_idxs) {
  std::unordered_map<int, int> frequency =
      GetClassFrequency(sample, sample.target_attributes_, sample_idxs);

  const double sample_gini = GiniIndex(frequency, sample_idxs.size());

  Node *rootNode =
      new Node(sample_gini, curr_depth, sample_idxs.size(), frequency);

  if (curr_depth == max_depth_ || sample_gini == 0.0 ||
      ShouldStop(frequency)) {
    rootNode->Classify();
    return rootNode;
  }

  int attribute_index;
  double gini_val;
  double attribute_value;

  // std::chrono::high_resolution_clock::time_point start =
  //     std::chrono::high_resolution_clock::now();

  std::tie(attribute_index, gini_val, attribute_value) =
      BestGini(sample, sample_idxs);

  // std::chrono::high_resolution_clock::time_point end =
  //     std::chrono::high_resolution_clock::now();
  // auto duration =
  //     std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
  //         .count();
  // std::cout << "Time computing gini " << " : " << duration / 1000.0
  //           << std::endl;  
  // split according to best gini
  // split dataset
  std::vector<int> left_idxs;
  std::vector<int> right_idxs;
  std::tie(left_idxs, right_idxs) =
      SplitSample(sample, attribute_value, attribute_index, sample_idxs);

  rootNode->feature_ = attribute_index;
  rootNode->splitted_value_ = attribute_value;
  rootNode->is_leaf_ = false;

  rootNode->left_child_ = Build(sample, curr_depth + 1, left_idxs);
  rootNode->right_child_ = Build(sample, curr_depth + 1, right_idxs);
  return rootNode;
}



int Predict(const std::vector<double> &query, Node *curr_node) { 
  if (curr_node->is_leaf_) {
    int clss;
    double prob;
    std::tie(clss, prob) = curr_node->class_label_;
    return clss;
  } else if(query[curr_node->feature_] <= curr_node->splitted_value_){
    return Predict(query, curr_node->left_child_);
  } else {
    return Predict(query, curr_node->right_child_);
  }
}
