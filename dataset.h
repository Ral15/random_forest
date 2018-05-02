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