#include <rclcpp/rclcpp.hpp>

#include "genie_sim_controllers/servo_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<genie_sim_controllers::ServoNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
