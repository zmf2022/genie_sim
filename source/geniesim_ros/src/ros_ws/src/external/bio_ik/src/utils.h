/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2016-2017, Philipp Sebastian Ruppel
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

#pragma once

#include <csignal>
#include <iostream>
#include <chrono>

#include <rclcpp/rclcpp.hpp>

#include <atomic>
#include <mutex>
#include <thread>
#include <typeindex>
#include <unordered_map>
#include <unordered_set>

#include <malloc.h>
#include <stdlib.h>

namespace bio_ik
{

inline double wallTime()
{
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

struct IKParams
{
  moveit::core::RobotModelConstPtr robot_model;
  const moveit::core::JointModelGroup * joint_model_group;
  std::string solver_class_name;
  bool enable_counter;
  int thread_count;
  int random_seed;
  double dpos;
  double drot;
  double dtwist;
  bool opt_no_wipeout;
  int population_size;
  int elite_count;
  bool linear_fitness;
};

#define LOG_STREAM (std::cerr)

template<class T> inline void vprint(std::ostream & s, const T & a) {s << a << std::endl;}
template<class T, class ... AA> inline void vprint(
  std::ostream & s, const T & a,
  const AA &... aa)
{
  s << a << " ";
  vprint(s, aa ...);
}

#define LOG2(...) vprint(LOG_STREAM, "ikbio ", __VA_ARGS__)

#ifdef ENABLE_LOG
#define LOG(...) LOG2(__VA_ARGS__)
#else
#define LOG(...)
#endif

#define LOG_VAR(v) LOG2(#v, (v));

#define LOG_FNC()

#define ERROR(...) \
  { \
    LOG2(__VA_ARGS__); \
    LOG_STREAM.flush(); \
    std::stringstream ss; \
    vprint(ss, __VA_ARGS__); \
    throw std::runtime_error(ss.str()); \
  }

#define FNPROFILER()
#define BLOCKPROFILER(name)
#define THREADPROFILER(name, id)
#define COUNTERPROFILER(name)

struct Profiler
{
  static void start() {}
};

__attribute__((always_inline)) inline double mix(double a, double b, double f)
{
  return a * (1.0 - f) + b * f;
}

__attribute__((always_inline)) inline double clamp(double v, double lo, double hi)
{
  if (v < lo) {v = lo;}
  if (v > hi) {v = hi;}
  return v;
}

__attribute__((always_inline)) inline double clamp2(double v, double lo, double hi)
{
  if (__builtin_expect(v < lo, 0)) {v = lo;}
  if (__builtin_expect(v > hi, 0)) {v = hi;}
  return v;
}

__attribute__((always_inline)) inline double smoothstep(float a, float b, float v)
{
  v = clamp((v - a) / (b - a), 0.0, 1.0);
  return v * v * (3.0 - 2.0 * v);
}

__attribute__((always_inline)) inline double sign(double f)
{
  if (f < 0.0) {f = -1.0;}
  if (f > 0.0) {f = +1.0;}
  return f;
}

template<class t>
class linear_int_distribution
{
  std::uniform_int_distribution<t> base;
  t n;

public:
  inline linear_int_distribution(t vrange)
  : n(vrange),
    base(0, vrange)
  {
  }
  template<class generator> inline t operator()(generator & g)
  {
    while (true) {
      t v = base(g) + base(g);
      if (v < n) {return n - v - 1;}
    }
  }
};

struct XORShift64
{
  uint64_t v;

public:
  XORShift64()
  : v(88172645463325252ull)
  {
  }
  __attribute__((always_inline)) inline uint64_t operator()()
  {
    v ^= v << 13;
    v ^= v >> 7;
    v ^= v << 17;
    return v;
  }
};

template<class BASE, class ... ARGS>
class Factory
{
  typedef BASE * (* Constructor)(ARGS...);
  struct ClassBase
  {
    std::string name;
    std::type_index type;
    virtual BASE * create(ARGS... args) const = 0;
    virtual BASE * clone(const BASE *) const = 0;
    ClassBase()
    : type(typeid(void))
    {
    }
  };
  typedef std::set<ClassBase *> MapType;
  static MapType & classes()
  {
    static MapType ff;
    return ff;
  }

public:
  template<class DERIVED>
  struct Class : ClassBase
  {
    BASE * create(ARGS... args) const {return new DERIVED(args ...);}
    BASE * clone(const BASE * o) const {return new DERIVED(*(const DERIVED *)o);}
    Class(const std::string & name)
    {
      this->name = name;
      this->type = typeid(DERIVED);
      classes().insert(this);
    }
    ~Class()
    {
      classes().erase(this);
    }
  };
  static BASE * create(const std::string & name, ARGS... args)
  {
    for (auto * f : classes()) {
      if (f->name == name) {return f->create(args ...);}}
    ERROR("class not found", name);
  }
  template<class DERIVED> static DERIVED * clone(const DERIVED * o)
  {
    for (auto * f : classes()) {
      if (f->type == typeid(*o)) {return (DERIVED *)f->clone(o);}}
    ERROR("class not found", typeid(*o).name());
  }
};

template<class T, size_t A>
struct aligned_allocator : public std::allocator<T>
{
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef T * pointer;
  typedef const T * const_pointer;
  typedef T & reference;
  typedef const T & const_reference;
  typedef T value_type;
  T * allocate(size_t s, const void * hint = 0)
  {
    void * p;
    if (posix_memalign(&p, A, sizeof(T) * s + 64)) {throw std::bad_alloc();}
    return (T *)p;
  }
  void deallocate(T * ptr, size_t s) {free(ptr);}
  template<class U>
  struct rebind
  {
    typedef aligned_allocator<U, A> other;
  };
};

template<class T>
struct aligned_vector : std::vector<T, aligned_allocator<T, 32>>
{
};

}
