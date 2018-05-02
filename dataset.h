#include <set>
#include <vector>

struct DataSet {
  std::vector<std::vector<double>> data;
  std::vector<int> target_values;
  std::set<int> target_attributes;
  std::vector<int> masked_attributes;
  int total_features;
  DataSet(){};
  DataSet(std::vector<std::vector<double>> d, std::vector<int> t_v,
          std::set<int> t_a, int t_f)
      : data(d),
        target_values(t_v),
        target_attributes(t_a),
        total_features(t_f){};
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


std::vector<std::set<double>> FeaturesAttributes(int num_of_features, int sample_size 
                                                 const std::vector<std::vector<double>> sample) {
  std::vector<std::set<double>> features_attributes(num_of_features);
  for (int i = 0;  i < num_of_features; i ++) {
    std::set<double> attributes;
    for (int j = 0; j < sample_size; j ++) {
      attributes.insert(sample[j][i]);
    }
    feature_attributes[i] = attributes;
  }
  return feature_attributes;
}


