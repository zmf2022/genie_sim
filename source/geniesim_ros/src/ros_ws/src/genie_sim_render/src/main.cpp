#include "genie_sim_render/render_node.hpp"
#include <rclcpp/rclcpp.hpp>

#include <cstdio>
#include <exception>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  std::shared_ptr<genie_sim_render::RenderNode> node;
  try {
    node = std::make_shared<genie_sim_render::RenderNode>();
  } catch (const std::exception & e) {
    std::fprintf(stderr, "[render_ovrtx] FATAL during RenderNode construction: %s\n", e.what());
    rclcpp::shutdown();
    return 1;
  } catch (...) {
    std::fprintf(
      stderr,
      "[render_ovrtx] FATAL: unknown exception during RenderNode construction\n");
    rclcpp::shutdown();
    return 1;
  }
  try {
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    std::fprintf(stderr, "[render_ovrtx] FATAL during spin: %s\n", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
