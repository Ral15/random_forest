#include <set>
#include <vector>


struct DataSet {
  std::vector<std::vector<double>> data_;
  std::vector<int> target_values_;
  std::set<int> target_attributes_;
  std::vector<int> masked_attributes_;
  int num_of_features_;
  DataSet(){};
  DataSet(std::vector<std::vector<double>> &d, std::vector<int> &t_v,
          std::set<int> &t_a, int t_f)
      : data_(d),
        target_values_(t_v),
        target_attributes_(t_a),
        num_of_features_(t_f){};
  ~DataSet(){};
};

std::vector<std::vector<double>> ReadSample(int num_data, int len_data) {
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
  return data;
}

std::tuple<std::vector<int>, std::set<int>> ReadTargetValues(int num_data) {
  // read target labels
  int y;
  std::vector<int> target_values;
  std::set<int> target_attributes;
  for (int i = 0; i < num_data; i++) {
    std::cin >> y;
    target_values.push_back(y);
    target_attributes.insert(y);
  }
  return std::make_tuple(target_values, target_attributes);
}

std::unordered_map<int, std::set<double>> FeatureAttributes(
    int num_of_features, int sample_size,
    const std::vector<std::vector<double>> &sample,
    const std::vector<int> &feature_idxs) {
  std::unordered_map<int, std::set<double>> feature_attributes;
  for (int i = 0; i < num_of_features; i++) {
    if (feature_idxs[i] == 1) {
      std::set<double> attributes;
      for (int j = 0; j < sample_size; j++) {
        attributes.insert(sample[j][i]);
      }
      feature_attributes[i] = attributes;
    }
  }
  return feature_attributes;
}
