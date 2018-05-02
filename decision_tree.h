#include <math.h>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "dataset.h"
#include "node.h"

using std::string;

struct DecisionTree {
  int id;
  Node *rootNode;
  int max_depth;
  DecisionTree(){};
  Node *Build(DataSet &d, int curr_depth);
  // void Predict(const std::vector<double> &q);
};

void PrintTabs(int tabs) {
  for (int i = 0; i < tabs; i++) std::cout << "|";
}

void PrintTree(Node *curr_node) {
  if (curr_node == nullptr) {
    return;
  }
  if (!curr_node->is_leaf) {
    PrintTabs(curr_node->curr_depth);
    std::cout << "X[" << curr_node->feature
              << "] <= " << curr_node->splitted_value << std::endl;
    PrintTabs(curr_node->curr_depth);
    std::cout << "gini = " << curr_node->gini_index << std::endl;
    PrintTabs(curr_node->curr_depth);
    std::cout << "samples = " << curr_node->sample_size << std::endl;
    PrintTabs(curr_node->curr_depth);
    std::cout << "values => ";
    for (auto it : curr_node->frequency) {
      std::cout << it.second << " ";
    }
    std::cout << std::endl;
    PrintTabs(curr_node->curr_depth);
    std::cout << "*Left*\n";
    PrintTree(curr_node->left_child);
    PrintTabs(curr_node->curr_depth);
    std::cout << "*Right*\n";
    PrintTree(curr_node->right_child);
  } else {
    PrintTabs(curr_node->curr_depth);
    std::cout << "gini = " << curr_node->gini_index << std::endl;
    PrintTabs(curr_node->curr_depth);
    std::cout << "samples = " << curr_node->sample_size << std::endl;
    PrintTabs(curr_node->curr_depth);
    std::cout << "values => ";
    for (auto it : curr_node->frequency) {
      std::cout << it.second << " ";
    }
    std::cout << std::endl;
    PrintTabs(curr_node->curr_depth);
    curr_node->Classify();
    // std::cout << "ENDDDDD" << std::endl;
    return;
  }
}

/*
  This is in case that the values for the class are different
*/
std::set<double> GetFeatureAttributes(
    const std::vector<std::vector<double>> &sample, int feature_index) {
  std::set<double> feature_attributes;
  for (int i = 0; i < sample.size(); i++) {
    feature_attributes.insert(sample[i][feature_index]);
  }
  return feature_attributes;
}

std::tuple<DataSet, DataSet> SplitSample(const DataSet &sample,
                                         const double split_value,
                                         const int idx_of_attribute) {
  std::tuple<DataSet, DataSet> splitted_sample;
  std::vector<std::vector<double>> left_sample;
  std::vector<std::vector<double>> right_sample;
  std::vector<int> left_target_values;
  std::vector<int> right_target_values;

  for (int i = 0; i < sample.data.size(); i++) {
    if (sample.data[i][idx_of_attribute] <= split_value) {
      left_sample.push_back(sample.data[i]);
      left_target_values.push_back(sample.target_values[i]);
    } else {
      right_sample.push_back(sample.data[i]);
      right_target_values.push_back(sample.target_values[i]);
    }
  }

  DataSet left = DataSet(left_sample, left_target_values, sample.target_attributes, sample.total_features);
  DataSet right = DataSet(right_sample, right_target_values, sample.target_attributes, sample.total_features);

  left.masked_attributes = right.masked_attributes = sample.masked_attributes;

  splitted_sample = std::make_tuple(left, right);
  return splitted_sample;
}

std::unordered_map<int, int> GetClassFrequency(const DataSet &sample,
                                               const std::set<int> &labels) {
  std::unordered_map<int, int> frequency;
  for (auto it : labels) {
    frequency[it];
  }
  for (int i = 0; i < sample.data.size(); i++) {
    frequency[sample.target_values[i]] += 1;
  }
  return frequency;
}

double GiniIndex(const std::unordered_map<int, int> frequency,
                 const int sample_size) {
  double gini_index = 0.0;
  for (auto it : frequency) {
    gini_index +=
        (double)it.second / sample_size * (1 - (double)it.second / sample_size);
  }
  return gini_index;
}

double GiniSplit(const DataSet &sample, const double feature_attribute,
                 const int attribute_index) {
  std::unordered_map<int, int> left_freq;
  std::unordered_map<int, int> right_freq;
  int l, r;
  l = r = 0;
  for (std::vector<int>::size_type i = 0; i < sample.data.size(); i++) {
    if (sample.data[i][attribute_index] <= feature_attribute) {
      left_freq[sample.target_values[i]] += 1;
      l++;
    } else {
      right_freq[sample.target_values[i]] += 1;
      r++;
    }
  }

  double left_gini =
      (((double)l / sample.data.size()) * GiniIndex(left_freq, l));
  double right_gini =
      (((double)r / sample.data.size()) * GiniIndex(right_freq, r));
  return left_gini + right_gini;
}

std::tuple<int, double, double> BestGini(
    const DataSet &sample) {
  double best_gini = std::numeric_limits<double>::max();
  double best_attr_indx = std::numeric_limits<double>::min();
  double attr_value_split = std::numeric_limits<double>::min();
  for (int i = 0; i < sample.total_features; i++) {
    if (sample.masked_attributes[i]) {
      std::set<double> feature_attributes =
          GetFeatureAttributes(sample.data, i);
      for (auto attr : feature_attributes) {  // compute the gini index
        double new_gini = GiniSplit(sample, attr, i);
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
  int freq_left = 0;
  for (auto it : frequency) {
    if (it.second > 0) freq_left = 1;
  }
  return freq_left;
}



Node *DecisionTree::Build(DataSet &sample, int curr_depth) {
  std::unordered_map<int, int> frequency =
      GetClassFrequency(sample, sample.target_attributes);

  const double sample_gini = GiniIndex(frequency, sample.data.size());

  Node *rootNode =
      new Node(sample_gini, curr_depth, sample.data.size(), frequency);

  if (curr_depth == max_depth || sample_gini == 0 ||
      ShouldStop(frequency) == 0) {
    rootNode->is_leaf = true;
    rootNode->Classify();
    return rootNode;
  }

  int attribute_index;
  double gini_val;
  double attribute_value;
  std::tie(attribute_index, gini_val, attribute_value) =
      BestGini(sample);
  // split according to best gini
  // split dataset
  DataSet left_data_set;
  DataSet right_data_set;
  std::tie(left_data_set, right_data_set) =
      SplitSample(sample, attribute_value, attribute_index);

  rootNode->feature = attribute_index;
  rootNode->splitted_value = attribute_value;
  rootNode->is_leaf = false;

  rootNode->left_child = Build(left_data_set, curr_depth + 1);
  rootNode->right_child = Build(right_data_set, curr_depth + 1);
  return rootNode;
}



int Predict(const std::vector<double> &query, Node *curr_node) { 
  if (curr_node->is_leaf) {
    // std::cout << "GANE" << std::endl;
    int clss;
    double prob;
    std::tie(clss, prob) = curr_node->class_label;
    // std::cout << "class label: " << clss << " prob: " << prob << std::endl;
    // std::cout << "Frequency of tree" << std::endl;
    // for (auto it: curr_node->frequency) {
    //   std::cout << it.first << ":" << it.second << " ";
    // }
    // std::cout << std::endl;
    return clss;
  } else if(query[curr_node->feature] <= curr_node->splitted_value){
    return Predict(query, curr_node->left_child);
  } else {
    return Predict(query, curr_node->right_child);
  }
}




