#pragma once

#include <Eigen/Dense>

namespace genie_sim_controllers::td
{

template<typename T, int InputDim, int Order>
class TrackingDifferentiator
{
public:
  TrackingDifferentiator(T dt, T r)
  : dt_(dt), r_(r)
  {
    static_assert(InputDim > 0, "InputDim must be greater than 0");
    static_assert(Order > 0, "Order must be greater than 0");
    x_.setZero();
    for (int i = 0; i <= Order; ++i) {
      coefficients_[i] = CombinationArray<Order>::value[i] * std::pow(r_, Order - i);
    }
  }

  void update(Eigen::Ref<const Eigen::Vector<T, InputDim>> input)
  {
    Eigen::Matrix<T, InputDim, Order> dx;
    dx.template leftCols<Order - 1>() = x_.template rightCols<Order - 1>();
    Eigen::Matrix<T, InputDim, Order> temp = -x_;
    temp.col(0) += input;
    dx.template rightCols<1>() = temp * coefficients_.template topRows<Order>();
    x_ += dx * dt_;
  }

  Eigen::Ref<const Eigen::Vector<T, InputDim>> x() const {return x_.col(0);}

  Eigen::Ref<const Eigen::Vector<T, InputDim>> dx() const
  {
    static_assert(Order > 1, "Order must be greater than 1");
    return x_.col(1);
  }

  Eigen::Ref<const Eigen::Vector<T, InputDim>> ddx() const
  {
    static_assert(Order > 2, "Order must be greater than 2");
    return x_.col(2);
  }

  void reset() {x_.setZero();}

  void setInitialState(Eigen::Ref<const Eigen::Vector<T, InputDim>> initial_state)
  {
    reset();
    x_.col(0) = initial_state;
  }

protected:
  template<int N, int K>
  struct Combination
  {
    static constexpr int value = Combination<N - 1, K - 1>::value + Combination<N - 1, K>::value;
  };

  template<int N>
  struct Combination<N, 0>
  {
    static constexpr int value = 1;
  };

  template<int N>
  struct Combination<N, N>
  {
    static constexpr int value = 1;
  };

  template<int N, int K>
  struct CombinationArrayHelper
  {
    static constexpr std::array<int, N + 1> compute()
    {
      static_assert(K <= N, "K must be less than or equal to N");
      auto arr = CombinationArrayHelper<N, K - 1>::compute();
      arr[K] = Combination<N, K>::value;
      return arr;
    }
  };

  template<int N>
  struct CombinationArrayHelper<N, 0>
  {
    static constexpr std::array<int, N + 1> compute()
    {
      std::array<int, N + 1> arr{};
      arr[0] = 1;
      return arr;
    }
  };

  template<int N>
  struct CombinationArray
  {
    static constexpr std::array<int, N + 1> value = CombinationArrayHelper<N, N>::compute();
  };

  T dt_;
  T r_;
  Eigen::Matrix<T, InputDim, Order> x_;
  Eigen::Vector<T, Order + 1> coefficients_;
};

}  // namespace genie_sim_controllers::td
