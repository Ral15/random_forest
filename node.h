#include <unordered_map>
#include <tuple>

struct Node {
  int feature;
  double gini_index;
  int curr_depth;
  double splitted_value;
  int sample_size;
  bool is_leaf;
  std::tuple<int, double> class_label;
  std::unordered_map<int, int> frequency;
  Node *left_child;
  Node *right_child;
  Node(double g_i, int c_d, int s_s, std::unordered_map<int, int> f)
      : gini_index(g_i), curr_depth(c_d), sample_size(s_s), frequency(f){};
  void Classify();
};

void Node::Classify() {
  // std::cout << "entre?" << std::endl;
  double prob_class = 0.0;
  int clss;
  // if (sample_size != 0) {
    for (auto it : frequency) {
      double curr_prob = (double)it.second / sample_size;
      if (prob_class < curr_prob) {
        prob_class = curr_prob;
        clss = it.first;
      }
    }
    // std::cout << "class => " << clss << " prob: " << prob_class * 100 << "%" << std::endl;
    class_label = std::make_tuple(clss, prob_class);   
  // }
}