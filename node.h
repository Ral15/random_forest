#include <tuple>
#include <unordered_map>

struct Node {
  int feature_;
  double gini_index_;
  int curr_depth_;
  double splitted_value_;
  int sample_size_;
  bool is_leaf_;
  std::tuple<int, double> class_label_;
  std::unordered_map<int, int> frequency_;
  Node *left_child_;
  Node *right_child_;
  Node(double g_i, int c_d, int s_s, std::unordered_map<int, int> &f)
      : gini_index_(g_i), curr_depth_(c_d), sample_size_(s_s), frequency_(f){};
  void Classify();
  ~Node() {
    delete left_child_;
    delete right_child_;
  }
};

void Node::Classify() {
  is_leaf_ = true;
  double prob_class = 0.0;
  int clss;
  for (auto &it : frequency_) {
    double curr_prob = (double)it.second / sample_size_;
    if (prob_class < curr_prob) {
      prob_class = curr_prob;
      clss = it.first;
    }
  }
  class_label_ = std::make_tuple(clss, prob_class);
}
