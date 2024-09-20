/*
 * cppgfx
 *
 * A tiny software rasterizer library for modern C++.
 *
 * https://github.com/cemderv/cppgfx
 *
 * Copyright (C) 2021-2024 Cemalettin Dervis
 *
 * Licensed under the MIT license (see file LICENSE.txt)
 *
 * Portions of the rasterizer are based on code from
 * https://github.com/trenki2/SoftwareRenderer by Markus Trenkwalder
 * licensed under the MIT license.
 *
 */

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ras
{
  using depth32f = float;

  template <typename T>
  struct red
  {
    red()
        : red(0)
    {
    }

    explicit constexpr red(T r)
        : r(r)
    {
    }

    T r;
  };

  template <typename T>
  struct rg
  {
    rg()
        : rg(T(0), T(0))
    {
    }

    explicit constexpr rg(T r, T g)
        : r(r)
        , g(g)
    {
    }

    T r;
    T g;
  };


  template <typename T>
  struct rgb
  {
    constexpr rgb()
        : rgb(T(0), T(0), T(0))
    {
    }

    constexpr rgb(T r, T g, T b)
        : r(r)
        , g(g)
        , b(b)
    {
    }

    T r;
    T g;
    T b;
  };

  template <typename T>
  struct rgba
  {
    constexpr rgba()
        : rgba(T(0), T(0), T(0), T(0))
    {
    }

    constexpr rgba(T r, T g, T b, T a)
        : r(r)
        , g(g)
        , b(b)
        , a(a)
    {
    }

    constexpr rgba(rgb<T> rgb, T a)
        : rgba(rgb.r, rgb.g, rgb.b, a)
    {
    }

    auto to_red() const -> red<T>
    {
      return red<T>{this->r};
    }

    auto to_rg() const -> rg<T>
    {
      return rg<T>{this->r, this->g};
    }

    auto to_rgb() const -> rgb<T>
    {
      return rgb<T>{this->r, this->g, this->b};
    }

    T r;
    T g;
    T b;
    T a;
  };

  using red32f  = red<float>;
  using rg32f   = rg<float>;
  using rgb32f  = rgb<float>;
  using rgba32f = rgba<float>;

  template <typename T>
  auto operator+(const rgb<T>& lhs, const rgb<T>& rhs) -> rgb<T>
  {
    return {
        lhs.r + rhs.r,
        lhs.g + rhs.g,
        lhs.b + rhs.b,
    };
  }

  template <typename T>
  auto operator-(const rgb<T>& lhs, const rgb<T>& rhs) -> rgb<T>
  {
    return {
        lhs.r - rhs.r,
        lhs.g - rhs.g,
        lhs.b - rhs.b,
    };
  }

  template <typename T>
  auto operator*(const rgb<T>& lhs, const rgb<T>& rhs) -> rgb<T>
  {
    return {
        lhs.r * rhs.r,
        lhs.g * rhs.g,
        lhs.b * rhs.b,
    };
  }

  template <typename T>
  auto operator*(const rgb<T>& lhs, float rhs) -> rgb<T>
  {
    return {
        lhs.r * rhs,
        lhs.g * rhs,
        lhs.b * rhs,
    };
  }

  template <typename T>
  auto operator*(const float lhs, const rgb<T>& rhs) -> rgb<T>
  {
    return rhs * lhs;
  }

  template <typename T>
  auto operator/(const rgb<T>& lhs, const rgb<T>& rhs) -> rgb<T>
  {
    return {
        lhs.r / rhs.r,
        lhs.g / rhs.g,
        lhs.b / rhs.b,
    };
  }

  template <typename T>
  auto operator/(const rgb<T>& lhs, float rhs) -> rgb<T>
  {
    return {
        lhs.r / rhs,
        lhs.r / rhs,
        lhs.r / rhs,
    };
  }

  namespace details
  {
    class vec4
    {
    public:
      vec4()
          : x(0.0F)
          , y(0.0F)
          , z(0.0F)
          , w(0.0F)
      {
      }

      vec4(float x, float y, float z, float w)
          : x(x)
          , y(y)
          , z(z)
          , w(w)
      {
      }

      template <typename T>
      explicit vec4(const T& vec4_like)
          : x(vec4_like.x)
          , y(vec4_like.y)
          , z(vec4_like.z)
          , w(vec4_like.w)
      {
      }

      template <typename T>
      void assign_to(T& vec4_like) const
      {
        vec4_like.x = this->x;
        vec4_like.y = this->y;
        vec4_like.z = this->z;
        vec4_like.w = this->w;
      }

      float x, y, z, w;
    };
  } // namespace details

  class r8g8b8a8
  {
  public:
    constexpr r8g8b8a8();

    constexpr r8g8b8a8(uint8_t r, uint8_t g, uint8_t b, uint8_t a);

    static auto from_rgbaf(const rgba32f& rgba) -> r8g8b8a8;

    auto to_rgbaf() const -> rgba32f;

    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
  };

  class b8g8r8a8
  {
  public:
    constexpr b8g8r8a8();

    constexpr b8g8r8a8(uint8_t b, uint8_t g, uint8_t r, uint8_t a);

    static auto from_rgba(const rgba32f& rgba) -> b8g8r8a8;

    auto to_rgbaf() const -> rgba32f;

    uint8_t b;
    uint8_t g;
    uint8_t r;
    uint8_t a;
  };

  class rect
  {
  public:
    constexpr rect();

    constexpr rect(int x, int y, unsigned width, unsigned height);

    int      x;
    int      y;
    unsigned width;
    unsigned height;
  };

  class viewport
  {
  public:
    constexpr viewport();

    constexpr explicit viewport(const rect& rect, float zmin = 0.0F, float zmax = 1.0F);

    rect  rect;
    float zmin;
    float zmax;
  };

  enum class primitive_topology
  {
    point = 1,
    line,
    triangle_list,
  };

  enum class cull_mode
  {
    none = 0,
    cull_clockwise,
    cull_counter_clockwise,
  };

  enum class image_address_mode
  {
    wrap = 1,
    clamp_to_edge,
    mirror,
  };

  enum class image_filter
  {
    linear = 1,
    point,
  };

  class sampler_state
  {
  public:
    image_filter       filter    = image_filter::linear;
    image_address_mode address_u = image_address_mode::mirror;
    image_address_mode address_v = image_address_mode::mirror;
    image_address_mode address_w = image_address_mode::mirror;
  };

  namespace sampler_states
  {
    static constexpr auto linear_clamp = sampler_state{
        .filter    = image_filter::linear,
        .address_u = image_address_mode::clamp_to_edge,
        .address_v = image_address_mode::clamp_to_edge,
        .address_w = image_address_mode::clamp_to_edge,
    };

    static constexpr auto linear_wrap = sampler_state{
        .filter    = image_filter::linear,
        .address_u = image_address_mode::wrap,
        .address_v = image_address_mode::wrap,
        .address_w = image_address_mode::wrap,
    };

    static constexpr auto point_clamp = sampler_state{
        .filter    = image_filter::point,
        .address_u = image_address_mode::clamp_to_edge,
        .address_v = image_address_mode::clamp_to_edge,
        .address_w = image_address_mode::clamp_to_edge,
    };

    static constexpr auto point_wrap = sampler_state{
        .filter    = image_filter::point,
        .address_u = image_address_mode::wrap,
        .address_v = image_address_mode::wrap,
        .address_w = image_address_mode::wrap,
    };
  } // namespace sampler_states

  enum class blend_op
  {
    // Result = (src_color * src_blend) + (dst_color * dst_blend)
    add = 1,

    // Result = (src_color * src_blend) - (dst_color * dst_blend)
    subtract = 2,

    // Result = (dst_color * dst_blend) - (src_color * src_blend)
    reverse_subtract = 3,

    // Result = min((src_color * src_blend), (dst_color * dst_blend))
    min = 4,

    // Result = max((src_color * src_blend), (dst_color * dst_blend))
    max = 5,
  };

  enum class blend
  {
    one                  = 1,
    zero                 = 2,
    src_color            = 3,
    inverse_src_color    = 4,
    src_alpha            = 5,
    inverse_src_alpha    = 6,
    dst_color            = 7,
    inverse_dst_color    = 8,
    dst_alpha            = 9,
    inv_dst_alpha        = 10,
    blend_factor         = 11,
    inverse_blend_factor = 12,
    src_alpha_saturation = 13,
  };

  enum class color_write_channels
  {
    none  = 1,
    red   = 2,
    green = 3,
    blue  = 4,
    alpha = 5,
    all   = 6,
  };

  class blend_state
  {
  public:
    bool                 is_blending_enabled = false;
    blend_op             blend_op_alpha      = blend_op::add;
    blend                dst_blend_alpha     = blend::zero;
    blend                src_blend_alpha     = blend::one;
    rgba32f              blend_factor        = {1.0F, 1.0F, 1.0F, 1.0F};
    blend_op             blend_op_rgb        = blend_op::add;
    blend                dst_blend_rgb       = blend::zero;
    blend                src_blend_rgb       = blend::one;
    color_write_channels write_channels      = color_write_channels::all;
  };

  namespace blend_states
  {
    static constexpr auto additive = blend_state{
        .is_blending_enabled = true,
        .blend_op_alpha      = blend_op::add,
        .dst_blend_alpha     = blend::one,
        .src_blend_alpha     = blend::src_alpha,
        .blend_factor        = {1.0F, 1.0F, 1.0F, 1.0F},
        .blend_op_rgb        = blend_op::add,
        .dst_blend_rgb       = blend::one,
        .src_blend_rgb       = blend::src_alpha,
        .write_channels      = color_write_channels::all,
    };

    static constexpr auto alpha_blend = blend_state{
        .is_blending_enabled = true,
        .blend_op_alpha      = blend_op::add,
        .dst_blend_alpha     = blend::inverse_src_alpha,
        .src_blend_alpha     = blend::one,
        .blend_factor        = {1.0F, 1.0F, 1.0F, 1.0F},
        .blend_op_rgb        = blend_op::add,
        .dst_blend_rgb       = blend::inverse_src_alpha,
        .src_blend_rgb       = blend::one,
        .write_channels      = color_write_channels::all,
    };

    static constexpr auto non_premultiplied = blend_state{
        .is_blending_enabled = true,
        .blend_op_alpha      = blend_op::add,
        .dst_blend_alpha     = blend::inverse_src_alpha,
        .src_blend_alpha     = blend::src_alpha,
        .blend_factor        = {1.0F, 1.0F, 1.0F, 1.0F},
        .blend_op_rgb        = blend_op::add,
        .dst_blend_rgb       = blend::inverse_src_alpha,
        .src_blend_rgb       = blend::src_alpha,
        .write_channels      = color_write_channels::all,
    };

    static constexpr auto opaque = blend_state{
        .is_blending_enabled = false,
        .blend_op_alpha      = blend_op::add,
        .dst_blend_alpha     = blend::zero,
        .src_blend_alpha     = blend::one,
        .blend_factor        = {1.0F, 1.0F, 1.0F, 1.0F},
        .blend_op_rgb        = blend_op::add,
        .dst_blend_rgb       = blend::zero,
        .src_blend_rgb       = blend::one,
        .write_channels      = color_write_channels::all,
    };
  } // namespace blend_states

  enum class compare_func
  {
    always = 1,
    never,
    less,
    less_equal,
    equal,
    greater_equal,
    greater,
    not_equal,
  };

  class depth_stencil_state
  {
  public:
    bool         is_depth_testing_enabled = true;
    bool         is_depth_write_enabled   = true;
    compare_func depth_test_func          = compare_func::less_equal;
  };

  namespace depth_stencil_states
  {
    static constexpr auto depth_default = depth_stencil_state{
        .is_depth_testing_enabled = true,
        .is_depth_write_enabled   = true,
        .depth_test_func          = compare_func::less_equal,
    };

    static constexpr auto depth_read = depth_stencil_state{
        .is_depth_testing_enabled = true,
        .is_depth_write_enabled   = false,
        .depth_test_func          = compare_func::less_equal,
    };

    static constexpr auto depth_none = depth_stencil_state{
        .is_depth_testing_enabled = false,
        .is_depth_write_enabled   = false,
        .depth_test_func          = compare_func::always,
    };
  } // namespace depth_stencil_states

  enum class fill_mode
  {
    solid,
  };

  class rasterizer_state
  {
  public:
    ras::cull_mode cull_mode               = cull_mode::none;
    ras::fill_mode fill_mode               = fill_mode::solid;
    bool           is_scissor_test_enabled = false;
    float          depth_bias              = 0.0f;
  };

  namespace rasterizer_states
  {
    static constexpr auto cull_none = rasterizer_state{
        .cull_mode               = cull_mode::none,
        .fill_mode               = fill_mode::solid,
        .is_scissor_test_enabled = false,
        .depth_bias              = 0.0f,
    };

    static constexpr auto cull_clockwise         = rasterizer_state();
    static constexpr auto cull_counter_clockwise = rasterizer_state();
  } // namespace rasterizer_states

  struct no_depth
  {
  };

  struct single_pixel_ps_input
  {
    unsigned x{};
    unsigned y{};
    float    u{};
    float    v{};
  };

  namespace details
  {
    template <typename T>
    auto decay_copy(T&&) -> std::decay_t<T>;

    template <typename T>
    concept has_xyzw = requires(T v) {
      {
        v.x
      } -> std::same_as<float>;
      {
        v.y
      } -> std::same_as<float>;
      {
        v.z
      } -> std::same_as<float>;
      {
        v.w
      } -> std::same_as<float>;
    };

    template <typename T>
    concept has_pos = requires(T v) { details::decay_copy(v.pos); };
  } // namespace details

  template <typename T>
  concept index_type = std::is_same_v<T, uint8_t> or std::is_same_v<T, uint16_t> or std::is_same_v<T, unsigned>;

  template <index_type I>
  static constexpr I no_index = static_cast<I>(-1);

  template <typename T>
  concept in_vertex = std::is_trivially_copyable_v<T>;

  /**
   * Represents the type of vertex that is returned from a vertex shader.
   */
  template <typename T>
  concept out_vertex = requires(T c) {
    std::is_standard_layout_v<T>;
    details::has_pos<T>and details::has_xyzw<T>;
  };

  template <typename Shader, typename InVertex, typename OutVertex>
  concept vertex_shader = out_vertex<OutVertex> and requires(Shader const s, InVertex i, OutVertex o) {
    {
      s.execute(i)
    } -> std::same_as<OutVertex>;
  };

  template <typename Shader, typename InVertex, size_t OutputCount>
  concept pixel_shader = out_vertex<InVertex> and requires(Shader const s, InVertex i) {
    {
      s.execute(i)
    } -> std::same_as<std::array<rgba32f, OutputCount>>;
  };

  template <typename Shader, size_t OutputCount>
  concept single_pixel_pixel_shader = requires(Shader const s, single_pixel_ps_input i) {
    {
      s.execute(i)
    } -> std::same_as<std::array<rgba32f, OutputCount>>;
  };

  namespace details
  {
    enum class vertex_clip_mask
    {
      pos_x = 0x01,
      neg_x = 0x02,
      pos_y = 0x04,
      neg_y = 0x08,
      pos_z = 0x10,
      neg_z = 0x20,
    };

    static inline auto operator|(vertex_clip_mask lhs, vertex_clip_mask rhs) -> vertex_clip_mask
    {
      return static_cast<vertex_clip_mask>(static_cast<uint8_t>(lhs) | static_cast<uint8_t>(rhs));
    }

    static inline auto operator&(vertex_clip_mask lhs, vertex_clip_mask rhs) -> bool
    {
      return static_cast<uint8_t>(lhs) & static_cast<uint8_t>(rhs);
    }

    class edge_equation
    {
    public:
      template <out_vertex OutVertex>
      edge_equation(const OutVertex& v0, const OutVertex& v1);

      auto evaluate(float x, float y) const -> float;

      auto test(float x, float y) const -> float;

      auto test(float xy) const -> float;

      auto step_x(float v, float step = 1.0F) const -> float;

      auto step_y(float v, float step = 1.0F) const -> float;

      float a;
      float b;
      float c;
      bool  tie;
    };

    class param_equation
    {
    public:
      param_equation();

      param_equation(float                p0,
                     float                p1,
                     float                p2,
                     const edge_equation& e0,
                     const edge_equation& e1,
                     const edge_equation& e2,
                     float                factor);

      auto evaluate(float x, float y) const -> float;

      auto step_x(float v, float step = 1.0F) const -> float;

      auto step_y(float v, float step = 1.0F) const -> float;

      float a;
      float b;
      float c;
    };

    template <out_vertex OutVertex>
    class out_vertex_traits
    {
    public:
      static_assert(sizeof(OutVertex) % sizeof(float) == 0, "output vertex must be dividable by float");

      static constexpr size_t num_floats  = sizeof(OutVertex) / sizeof(float);
      static constexpr size_t num_attribs = num_floats - 4;

      static auto float_ptr(OutVertex& v) -> float*
      {
        return reinterpret_cast<float*>(&v);
      }

      static auto float_ptr(const OutVertex& v) -> const float*
      {
        return reinterpret_cast<const float*>(&v);
      }

      static auto attrib_ptr(OutVertex& v) -> float*
      {
        return float_ptr(v) + 4;
      }

      static auto attrib_ptr(const OutVertex& v) -> const float*
      {
        return float_ptr(v) + 4;
      }
    };

    template <out_vertex T>
    class triangle_equations
    {
    public:
      triangle_equations(const T& v0, const T& v1, const T& v2);

      edge_equation                                                 e0;
      edge_equation                                                 e1;
      edge_equation                                                 e2;
      float                                                         area2;
      param_equation                                                z;
      param_equation                                                inv_w;
      std::array<param_equation, out_vertex_traits<T>::num_attribs> attribs;
    };

    template <index_type I, size_t Size>
    class vertex_cache
    {
    public:
      vertex_cache();

      void clear();

      void set(I in_index, I out_index);

      auto lookup(I in_index) const -> I;

    private:
      std::array<I, Size> m_input_index;
      std::array<I, Size> m_output_index;
    };

    class edge_data
    {
    public:
      template <out_vertex T>
      edge_data(const triangle_equations<T>& eqn, float x, float y);

      template <out_vertex T>
      void step_x(const triangle_equations<T>& eqn, float step);

      template <out_vertex T>
      void step_y(const triangle_equations<T>& eqn, float step);

      template <out_vertex T>
      auto test(const triangle_equations<T>& eqn) const -> bool;

      float ev0;
      float ev1;
      float ev2;
    };

    template <out_vertex OutVertex>
    class interpolated_vertex
    {
    public:
      interpolated_vertex(const triangle_equations<OutVertex>& eqn, float x, float y);

      explicit interpolated_vertex(const OutVertex& v);

      void step_x(const triangle_equations<OutVertex>& eqn);

      void step_y(const triangle_equations<OutVertex>& eqn);

      auto to_out_vertex() const -> OutVertex;

      vec4                                                         pos;
      float                                                        invW;
      std::array<float, out_vertex_traits<OutVertex>::num_attribs> tmpAttribs;
      std::array<float, out_vertex_traits<OutVertex>::num_attribs> attribs;
    };

    template <index_type I>
    class poly_clipper
    {
    public:
      explicit poly_clipper(size_t initial_capacity);

      void clear();

      void init(I i1, I i2, I i3);

      template <out_vertex T>
      void clip_to_plane(std::vector<T>& vertices, float a, float b, float c, float d);

      auto indices() const -> const std::vector<I>&;

      auto is_fully_clipped() const -> bool;

    private:
      std::vector<I> m_indices_in;
      std::vector<I> m_indices_out;
    };

    class line_clipper
    {
    public:
      template <out_vertex T>
      line_clipper(const T& v0, const T& v1);

      void clip_to_plane(float a, float b, float c, float d);

      vec4  v0            = {};
      vec4  v1            = {};
      float t0            = 0.0F;
      float t1            = 0.0F;
      bool  fully_clipped = false;
    };

    auto tex_coord_to_index(float coord, unsigned size, image_address_mode mode) -> unsigned;

    auto tex_coord_to_index_absolute(float coord, float size, image_address_mode mode) -> float;
  } // namespace details

  template <in_vertex T>
  class vertex_view
  {
  public:
    vertex_view(const T* data, size_t size)
        : m_data(data)
        , m_size(size)
    {
    }

    vertex_view(const std::vector<T>& vec)
        : m_data(vec.data())
        , m_size(vec.size())
    {
    }

    template <size_t Size>
    vertex_view(const std::array<T, Size>& arr)
        : m_data(arr.data())
        , m_size(arr.size())
    {
    }

    vertex_view(std::span<T>& span)
        : m_data(span.data())
        , m_size(span.size())
    {
    }

    vertex_view(std::span<const T>& span)
        : m_data(span.data())
        , m_size(span.size())
    {
    }

    vertex_view(const std::initializer_list<T>& list)
        : m_data(list.begin())
        , m_size(list.size())
    {
    }

    auto data() const -> const T*
    {
      return m_data;
    }

    auto size() const -> size_t
    {
      return m_size;
    }

    auto begin() const -> const T*
    {
      return m_data;
    }

    auto end() const -> const T*
    {
      return m_data + m_size;
    }

    auto operator[](size_t index) const -> const T&
    {
      return m_data[index];
    }

  private:
    const T* m_data;
    size_t   m_size;
  };

  template <index_type T>
  class index_view
  {
  public:
    index_view(const T* data, size_t size)
        : m_data(data)
        , m_size(size)
    {
    }

    index_view(const std::vector<T>& vec)
        : m_data(vec.data())
        , m_size(vec.size())
    {
    }

    template <size_t Size>
    index_view(const std::array<T, Size>& arr)
        : m_data(arr.data())
        , m_size(arr.size())
    {
    }

    index_view(std::span<T>& span)
        : m_data(span.data())
        , m_size(span.size())
    {
    }

    index_view(std::span<const T>& span)
        : m_data(span.data())
        , m_size(span.size())
    {
    }

    index_view(const std::initializer_list<T>& list)
        : m_data(list.begin())
        , m_size(list.size())
    {
    }

    auto data() const -> const T*
    {
      return m_data;
    }

    auto size() const -> size_t
    {
      return m_size;
    }

    auto begin() const -> const T*
    {
      return m_data;
    }

    auto end() const -> const T*
    {
      return m_data + m_size;
    }

    auto operator[](size_t index) const -> const T&
    {
      return m_data[index];
    }

  private:
    const T* m_data;
    size_t   m_size;
  };

  template <out_vertex OutVertex, index_type IndexType = uint16_t, size_t VertexCacheSize = 64U>
  class raster_cache
  {
  public:
    explicit raster_cache(size_t initial_capacity = 256)
        : pclipper(initial_capacity)
    {
      vertices_out.reserve(initial_capacity);
      indices_out.reserve(initial_capacity);
      clip_mask.reserve(8);
      already_processed.reserve(16);
    }

    void clear()
    {
      pclipper.clear();
      vertices_out.clear();
      indices_out.clear();
      vcache.clear();
      clip_mask.clear();
      already_processed.clear();
    }

    details::poly_clipper<IndexType>                  pclipper;
    std::vector<OutVertex>                            vertices_out;
    std::vector<IndexType>                            indices_out;
    details::vertex_cache<IndexType, VertexCacheSize> vcache;
    std::vector<uint8_t>                              clip_mask;
    std::vector<bool>                                 already_processed;
  };

  template <typename T>
  concept image_format =
      std::is_default_constructible_v<T> and std::is_copy_constructible_v<T> and std::is_copy_assignable_v<T>;

  template <typename T>
  concept depth_format = image_format<T> and (std::is_same_v<T, depth32f> or std::is_same_v<T, no_depth>);

  template <typename T>
  concept color_format = image_format<T> and not depth_format<T>;

  class image
  {
  protected:
    image();

    image(unsigned width, unsigned height, unsigned depth);

  public:
    image(const image& rhs);

    auto operator=(const image& rhs) -> image&;

    image(image&&) noexcept;

    auto operator=(image&& rhs) noexcept -> image&;

    ~image() noexcept;

    auto width() const -> unsigned;

    auto height() const -> unsigned;

    auto depth() const -> unsigned;

    auto widthf() const -> float;

    auto heightf() const -> float;

    auto depthf() const -> float;

  private:
    unsigned m_width;
    unsigned m_height;
    unsigned m_depth;
    float    m_widthf;
    float    m_heightf;
    float    m_depthf;
  };

  template <image_format Format>
  class image2d : public image
  {
  public:
    using format_t = Format;

    image2d();

    image2d(unsigned width, unsigned height);

    image2d(unsigned width, unsigned height, Format* foreign_data);

    image2d(const image2d&) = delete;

    void operator=(const image2d&) = delete;

    image2d(image2d&&) noexcept;

    auto operator=(image2d&&) noexcept -> image2d&;

    ~image2d() noexcept;

    void clear(const Format& value);

    auto get(unsigned x, unsigned y) const -> const Format&;

    void set(unsigned x, unsigned y, const Format& value);

    auto operator[](unsigned index) -> Format&;

    auto operator[](unsigned index) const -> const Format&;

    auto data() -> Format*;

    auto data() const -> const Format*;

    auto data_span() -> std::span<Format>;

    auto data_span() const -> std::span<const Format>;

    auto pixel_count() const -> unsigned;

    auto size_in_bytes() const -> size_t;

    auto row_pitch() const -> size_t;

    auto aspect_ratio() const -> float;

    auto owns_data() const -> bool;

  private:
    void destroy();

    Format* m_data;
    bool    m_owns_data;
  };

  using image2d_rgba    = image2d<r8g8b8a8>;
  using image2d_bgra    = image2d<b8g8r8a8>;
  using image2d_depth32 = image2d<depth32f>;

  template <color_format Format>
  auto read_image_color(const image2d<Format>& image, unsigned x, unsigned y) -> rgba32f;

  template <color_format Format>
  void put_image_color(image2d<Format>& image, unsigned x, unsigned y, const rgba32f& color);

  template <>
  auto read_image_color<r8g8b8a8>(const image2d<r8g8b8a8>& image, unsigned x, unsigned y) -> rgba32f;

  template <>
  void put_image_color<r8g8b8a8>(image2d<r8g8b8a8>& image, unsigned x, unsigned y, const rgba32f& color);

  template <>
  auto read_image_color<b8g8r8a8>(const image2d<b8g8r8a8>& image, unsigned x, unsigned y) -> rgba32f;

  template <>
  void put_image_color<b8g8r8a8>(image2d<b8g8r8a8>& image, unsigned x, unsigned y, const rgba32f& color);

  template <>
  auto read_image_color<rgba32f>(const image2d<rgba32f>& image, unsigned x, unsigned y) -> rgba32f;

  template <>
  void put_image_color<rgba32f>(image2d<rgba32f>& image, unsigned x, unsigned y, const rgba32f& color);

  template <>
  auto read_image_color<rgb32f>(const image2d<rgb32f>& image, unsigned x, unsigned y) -> rgba32f;

  template <>
  void put_image_color<rgb32f>(image2d<rgb32f>& image, unsigned x, unsigned y, const rgba32f& color);

  template <>
  auto read_image_color<rg32f>(const image2d<rg32f>& image, unsigned x, unsigned y) -> rgba32f;

  template <>
  void put_image_color<rg32f>(image2d<rg32f>& image, unsigned x, unsigned y, const rgba32f& color);

  template <>
  auto read_image_color<red32f>(const image2d<red32f>& image, unsigned x, unsigned y) -> rgba32f;

  template <>
  void put_image_color<red32f>(image2d<red32f>& image, unsigned x, unsigned y, const rgba32f& color);

  template <image_format Format>
  auto sample(const image2d<Format>& image, float u, float v, const sampler_state& sampler) -> rgba32f;

  template <image_format DestinationFormat, image_format SourceFormat>
  auto convert_image(const image2d<SourceFormat>& source_image) -> image2d<DestinationFormat>;

  template <typename... Images>
  class color_images
  {
  public:
    static constexpr size_t count = sizeof...(Images);

    explicit color_images(Images&... as)
        : m_width(0)
        , m_height(0)
        , m_widthf(0)
        , m_heightf(0)
        , m_images(std::forward_as_tuple(as...))
    {
      verify_image(m_images);
    }

    auto width() const -> unsigned
    {
      return m_width;
    }

    auto height() const -> unsigned
    {
      return m_height;
    }

    auto widthf() const -> float
    {
      return m_widthf;
    }

    auto heightf() const -> float
    {
      return m_heightf;
    }

    auto images() -> std::tuple<Images&...>&
    {
      return m_images;
    }

  private:
    template <size_t I = 0, typename... Ts>
    constexpr void verify_image(std::tuple<Ts...>& images_tuple)
    {
      if constexpr (I != sizeof...(Ts))
      {
        const auto& image = std::get<I>(images_tuple);

        // Verify that the image has some memory/data allocated.
        assert(image.data() != nullptr);

        if constexpr (I == 0)
        {
          m_width   = image.width();
          m_height  = image.height();
          m_widthf  = image.widthf();
          m_heightf = image.heightf();
        }

        verify_image<I + 1>(images_tuple);
      }
    }

    unsigned               m_width;
    unsigned               m_height;
    float                  m_widthf;
    float                  m_heightf;
    std::tuple<Images&...> m_images;
  };

  template <in_vertex  InVertex,
            out_vertex OutVertex,
            index_type IndexType,
            typename ColorImages,
            depth_format                                DepthFormat,
            vertex_shader<InVertex, OutVertex>          VertexShader,
            pixel_shader<OutVertex, ColorImages::count> PixelShader,
            size_t                                      VertexCacheSize>
  void draw_indexed(ColorImages                                          color_targets,
                    image2d<DepthFormat>&                                depth_target,
                    const VertexShader&                                  vertex_shader,
                    const PixelShader&                                   pixel_shader,
                    const viewport&                                      viewport,
                    const rect&                                          scissor_rect,
                    const blend_state&                                   blend_state,
                    const depth_stencil_state&                           depth_stencil_state,
                    const rasterizer_state&                              rasterizer_state,
                    primitive_topology                                   topology,
                    raster_cache<OutVertex, IndexType, VertexCacheSize>& cache,
                    vertex_view<InVertex>                                vertices,
                    index_view<IndexType>                                indices,
                    size_t                                               start_index,
                    size_t                                               index_count,
                    int                                                  base_vertex_index = 0);

  enum class uv_origin
  {
    top_left = 1,
    bottom_left,
  };

  template <typename ColorImages, single_pixel_pixel_shader<ColorImages::count> PixelShader>
  void for_each_pixel(ColorImages        color_targets,
                      const PixelShader& pixel_shader,
                      const blend_state& blend_state,
                      ras::uv_origin     uv_origin = uv_origin::top_left);

  auto smoothstep(float start, float end, float t) -> float;

  auto smootherstep(float start, float end, float t) -> float;

  auto step(float y, float x) -> float;

  template <typename T>
  static inline auto sign(T val) -> int
  {
    return (T(0) < val) - (val < T(0));
  }

  // ----------------------
  // IMPLEMENTATION
  // ----------------------

  inline constexpr r8g8b8a8::r8g8b8a8()
      : r(0)
      , g(0)
      , b(0)
      , a(0)
  {
  }

  inline constexpr r8g8b8a8::r8g8b8a8(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
      : r(r)
      , g(g)
      , b(b)
      , a(a)
  {
  }

  inline auto r8g8b8a8::from_rgbaf(const rgba32f& rgba) -> r8g8b8a8
  {
    return {
        static_cast<uint8_t>(std::clamp(rgba.r * 255.0F, 0.0F, 255.0F)),
        static_cast<uint8_t>(std::clamp(rgba.g * 255.0F, 0.0F, 255.0F)),
        static_cast<uint8_t>(std::clamp(rgba.b * 255.0F, 0.0F, 255.0F)),
        static_cast<uint8_t>(std::clamp(rgba.a * 255.0F, 0.0F, 255.0F)),
    };
  }

  inline auto r8g8b8a8::to_rgbaf() const -> rgba32f
  {
    constexpr float inv = 1.0F / 255.0f;
    return {
        static_cast<float>(r) * inv,
        static_cast<float>(g) * inv,
        static_cast<float>(b) * inv,
        static_cast<float>(a) * inv,
    };
  }

  inline constexpr b8g8r8a8::b8g8r8a8()
      : b(0)
      , g(0)
      , r(0)
      , a(0)
  {
  }

  inline constexpr b8g8r8a8::b8g8r8a8(uint8_t b, uint8_t g, uint8_t r, uint8_t a)
      : b(b)
      , g(g)
      , r(r)
      , a(a)
  {
  }

  inline auto b8g8r8a8::from_rgba(const rgba32f& rgba) -> b8g8r8a8
  {
    return {
        static_cast<uint8_t>(std::clamp(rgba.b * 255.0F, 0.0F, 255.0F)),
        static_cast<uint8_t>(std::clamp(rgba.g * 255.0F, 0.0F, 255.0F)),
        static_cast<uint8_t>(std::clamp(rgba.r * 255.0F, 0.0F, 255.0F)),
        static_cast<uint8_t>(std::clamp(rgba.a * 255.0F, 0.0F, 255.0F)),
    };
  }

  inline auto b8g8r8a8::to_rgbaf() const -> rgba32f
  {
    constexpr float inv = 1.0F / 255.0f;

    return {
        static_cast<float>(r) * inv,
        static_cast<float>(g) * inv,
        static_cast<float>(b) * inv,
        static_cast<float>(a) * inv,
    };
  }

  namespace details
  {
    static inline auto tex_index_2d(unsigned x, unsigned y, unsigned width) -> unsigned
    {
      return y * width + x;
    }

    static inline auto texcoord_to_index(float coord, float size, image_address_mode mode) -> float
    {
      switch (mode)
      {
        case image_address_mode::wrap: return std::abs(std::fmod(coord, size));
        case image_address_mode::clamp_to_edge: {
          return std::clamp(coord, 0.0F, size - 1.0F);
        }
        case image_address_mode::mirror: {
          const auto modfn             = [](float a, float n) { return std::fmod((std::fmod(a, n) + n), n); };
          const auto mirrorfn          = [](float a) { return a >= 0.0F ? a : -(1.0F + a); };
          const auto mirrored_repeatfn = [&](float x) {
            return (size - 1.0F) - mirrorfn(modfn(x, 2.0F * size) - size);
          };

          return mirrored_repeatfn(coord);
        }
      }

      return 0.0F;
    }

    /// c01 --- c11
    ///  |       |
    ///  |       |
    /// c00 --- c10
    static inline auto blerp(float c00, float c10, float c01, float c11, float tx, float ty) -> float
    {
      const auto a = std::lerp(c00, c10, tx);
      const auto b = std::lerp(c01, c11, tx);
      return std::lerp(a, b, ty);
    }
  } // namespace details

  template <color_format Format>
  inline auto read_image_color(const image2d<Format>& image, unsigned x, unsigned y) -> rgba32f
  {
    (void)image;
    (void)x;
    (void)y;
    return {};
  }

  template <color_format Format>
  inline void put_image_color(image2d<Format>& image, unsigned x, unsigned y, const rgba32f& color)
  {
    (void)image;
    (void)x;
    (void)y;
    (void)color;
  }

  // R8G8B8A8 image read/write
  template <>
  inline auto read_image_color<r8g8b8a8>(const image2d<r8g8b8a8>& image, unsigned x, unsigned y) -> rgba32f
  {
    return image.get(x, y).to_rgbaf();
  }

  template <>
  inline void put_image_color<r8g8b8a8>(image2d<r8g8b8a8>& image, unsigned x, unsigned y, const rgba32f& color)
  {
    image.set(x, y, r8g8b8a8::from_rgbaf(color));
  }

  // B8G8R8A8 image read/write
  template <>
  inline auto read_image_color<b8g8r8a8>(const image2d<b8g8r8a8>& image, unsigned x, unsigned y) -> rgba32f
  {
    return image.get(x, y).to_rgbaf();
  }

  template <>
  inline void put_image_color<b8g8r8a8>(image2d<b8g8r8a8>& image, unsigned x, unsigned y, const rgba32f& color)
  {
    image.set(x, y, b8g8r8a8::from_rgba(color));
  }

  // rgba32f image read/write
  template <>
  inline auto read_image_color<rgba32f>(const image2d<rgba32f>& image, unsigned x, unsigned y) -> rgba32f
  {
    return image.get(x, y);
  }

  template <>
  inline void put_image_color<rgba32f>(image2d<rgba32f>& image, unsigned x, unsigned y, const rgba32f& color)
  {
    image.set(x, y, color);
  }

  // rgb32f image read/write
  template <>
  inline auto read_image_color<rgb32f>(const image2d<rgb32f>& image, unsigned x, unsigned y) -> rgba32f
  {
    return rgba32f(image.get(x, y), 1.0F);
  }

  template <>
  inline void put_image_color<rgb32f>(image2d<rgb32f>& image, unsigned x, unsigned y, const rgba32f& color)
  {
    image.set(x, y, color.to_rgb());
  }

  // RG32f image read/write
  template <>
  inline auto read_image_color<rg32f>(const image2d<rg32f>& image, unsigned x, unsigned y) -> rgba32f
  {
    const rg32f rgb = image.get(x, y);

    return {rgb.r, rgb.g, 0.0F, 1.0F};
  }

  template <>
  inline void put_image_color<rg32f>(image2d<rg32f>& image, unsigned x, unsigned y, const rgba32f& color)
  {
    image.set(x, y, color.to_rg());
  }

  // Red32f image read/write
  template <>
  inline auto read_image_color<red32f>(const image2d<red32f>& image, unsigned x, unsigned y) -> rgba32f
  {
    return {image.get(x, y).r, 0.0F, 0.0F, 1.0F};
  }

  template <>
  inline void put_image_color<red32f>(image2d<red32f>& image, unsigned x, unsigned y, const rgba32f& color)
  {
    image.set(x, y, color.to_red());
  }

  template <depth_format Format>
  inline auto read_image_depth(const image2d<Format>& image, unsigned x, unsigned y) -> float
  {
    (void)image;
    (void)x;
    (void)y;
    return 0.0F;
  }

  template <depth_format Format>
  inline void put_image_depth(image2d<Format>& image, unsigned x, unsigned y, float depth)
  {
    (void)image;
    (void)x;
    (void)y;
    (void)depth;
  }

  template <>
  inline auto read_image_depth<depth32f>(const image2d<depth32f>& image, unsigned x, unsigned y) -> float
  {
    return image.get(x, y);
  }

  template <>
  inline void put_image_depth<depth32f>(image2d<depth32f>& image, unsigned x, unsigned y, float depth)
  {
    image.set(x, y, depth);
  }

  template <image_format Format>
  inline auto sample(const image2d<Format>& image, float u, float v, const sampler_state& sampler) -> rgba32f
  {
    const auto either_color_or_depth = [&](unsigned x, unsigned y) {
      if constexpr (depth_format<Format>)
      {
        const auto depth = read_image_depth(image, x, y);
        return rgba32f(depth, depth, depth, 1.0F);
      }
      else if constexpr (color_format<Format>)
      {
        return read_image_color(image, x, y);
      }
    };

    const auto w = image.width();
    const auto h = image.height();

    const auto uv_to_xy = [&](float xf, float yf) {
      const auto x = details::texcoord_to_index(xf, float(w), sampler.address_u);
      const auto y = details::texcoord_to_index(yf, float(h), sampler.address_v);
      return std::make_pair(x, y);
    };

    switch (sampler.filter)
    {
      case image_filter::linear: {
        const auto [xCoord, yCoord] = uv_to_xy(u * float(w), v * float(h));

        const auto x_floor = std::floor(xCoord);
        const auto y_floor = std::floor(yCoord);

        const auto x_ceil = std::ceil(xCoord);
        const auto y_ceil = std::ceil(yCoord);

        const auto tx = xCoord - x_floor;
        const auto ty = yCoord - y_floor;

        const auto c00 = [&] {
          const auto [x, y] = uv_to_xy(x_floor, y_floor);
          return either_color_or_depth(unsigned(x), unsigned(y));
        }();

        const auto c10 = [&] {
          const auto [x, y] = uv_to_xy(x_ceil, y_floor);
          return either_color_or_depth(unsigned(x), unsigned(y));
        }();

        const auto c01 = [&] {
          const auto [x, y] = uv_to_xy(x_floor, y_ceil);
          return either_color_or_depth(unsigned(x), unsigned(y));
        }();

        const auto c11 = [&] {
          const auto [x, y] = uv_to_xy(x_ceil, y_ceil);
          return either_color_or_depth(unsigned(x), unsigned(y));
        }();

        const auto red   = details::blerp(c00.r, c10.r, c01.r, c11.r, tx, ty);
        const auto green = details::blerp(c00.g, c10.g, c01.g, c11.g, tx, ty);
        const auto blue  = details::blerp(c00.b, c10.b, c01.b, c11.b, tx, ty);
        const auto alpha = details::blerp(c00.a, c10.a, c01.a, c11.a, tx, ty);

        return {red, green, blue, alpha};
      }
      case image_filter::point: {
        const auto [x_coord, y_coord] = uv_to_xy(u * float(w), v * float(h));
        const auto x                  = std::floor(x_coord);
        const auto y                  = std::floor(y_coord);

        return either_color_or_depth(unsigned(x), unsigned(y));
      }
    }

    return {1, 0, 0, 1};
  }

  template <>
  inline auto convert_image<r8g8b8a8>(const image2d<b8g8r8a8>& source_image) -> image2d<r8g8b8a8>
  {
    image2d<r8g8b8a8> ret{source_image.width(), source_image.height()};
    const auto        pixel_count = ret.pixel_count();
    r8g8b8a8*         dst         = ret.data();
    const b8g8r8a8*   src         = source_image.data();
    for (unsigned i = 0; i < pixel_count; ++i, ++dst, ++src)
    {
      dst->r = src->r;
      dst->g = src->g;
      dst->b = src->b;
      dst->a = src->a;
    }
    return ret;
  }

  template <>
  inline auto convert_image<b8g8r8a8>(const image2d<r8g8b8a8>& source_image) -> image2d<b8g8r8a8>
  {
    image2d<b8g8r8a8> ret{source_image.width(), source_image.height()};
    const auto        pixel_count = ret.pixel_count();
    b8g8r8a8*         dst         = ret.data();
    const r8g8b8a8*   src         = source_image.data();
    for (unsigned i = 0; i < pixel_count; ++i, ++dst, ++src)
    {
      dst->r = src->r;
      dst->g = src->g;
      dst->b = src->b;
      dst->a = src->a;
    }
    return ret;
  }

  template <>
  inline auto convert_image<r8g8b8a8>(const image2d<rgb32f>& source_image) -> image2d<r8g8b8a8>
  {
    auto          ret         = image2d<r8g8b8a8>(source_image.width(), source_image.height());
    const auto    pixel_count = ret.pixel_count();
    auto          dst         = ret.data();
    const rgb32f* src         = source_image.data();

    for (unsigned i = 0; i < pixel_count; ++i, ++dst, ++src)
    {
      dst->r = static_cast<uint8_t>(std::clamp(src->r * 255.0F, 0.0F, 255.0F));
      dst->g = static_cast<uint8_t>(std::clamp(src->g * 255.0F, 0.0F, 255.0F));
      dst->b = static_cast<uint8_t>(std::clamp(src->b * 255.0F, 0.0F, 255.0F));
      dst->a = static_cast<uint8_t>(255);
    }

    return ret;
  }

  template <>
  inline auto convert_image<b8g8r8a8>(const image2d<rgb32f>& source_image) -> image2d<b8g8r8a8>
  {
    auto          ret         = image2d<b8g8r8a8>(source_image.width(), source_image.height());
    const auto    pixel_count = ret.pixel_count();
    auto          dst         = ret.data();
    const rgb32f* src         = source_image.data();

    for (unsigned i = 0; i < pixel_count; ++i, ++dst, ++src)
    {
      dst->b = static_cast<uint8_t>(std::clamp(src->r * 255.0F, 0.0F, 255.0F));
      dst->g = static_cast<uint8_t>(std::clamp(src->g * 255.0F, 0.0F, 255.0F));
      dst->r = static_cast<uint8_t>(std::clamp(src->b * 255.0F, 0.0F, 255.0F));
      dst->a = static_cast<uint8_t>(255);
    }

    return ret;
  }

  template <>
  inline auto convert_image<r8g8b8a8>(const image2d<depth32f>& source_image) -> image2d<r8g8b8a8>
  {
    auto            ret         = image2d<r8g8b8a8>(source_image.width(), source_image.height());
    const auto      pixel_count = ret.pixel_count();
    r8g8b8a8*       dst         = ret.data();
    const depth32f* src         = source_image.data();

    for (unsigned i = 0; i < pixel_count; ++i, ++dst, ++src)
    {
      const auto d = static_cast<uint8_t>(std::clamp(*src * 255.0F, 0.0F, 255.0F));
      dst->r       = d;
      dst->g       = d;
      dst->b       = d;
      dst->a       = static_cast<uint8_t>(255);
    }
    return ret;
  }

  template <>
  inline auto convert_image<b8g8r8a8>(const image2d<depth32f>& source_image) -> image2d<b8g8r8a8>
  {
    image2d<b8g8r8a8> ret{source_image.width(), source_image.height()};
    const auto        pixel_count = ret.pixel_count();
    b8g8r8a8*         dst         = ret.data();
    const depth32f*   src         = source_image.data();
    for (unsigned i = 0; i < pixel_count; ++i, ++dst, ++src)
    {
      const auto d = static_cast<uint8_t>(std::clamp(*src * 255.0F, 0.0F, 255.0F));
      dst->b       = d;
      dst->g       = d;
      dst->r       = d;
      dst->a       = static_cast<uint8_t>(255);
    }
    return ret;
  }

  constexpr inline rect::rect()
      : x(0)
      , y(0)
      , width(0U)
      , height(0U)
  {
  }

  constexpr inline rect::rect(int x, int y, unsigned width, unsigned height)
      : x(x)
      , y(y)
      , width(width)
      , height(height)
  {
  }

  constexpr inline viewport::viewport()
      : rect()
      , zmin(0.0F)
      , zmax(1.0F)
  {
  }

  constexpr inline viewport::viewport(const ras::rect& rect, float zmin, float zmax)
      : rect(rect)
      , zmin(zmin)
      , zmax(zmax)
  {
  }

  inline auto image::width() const -> unsigned
  {
    return m_width;
  }

  inline auto image::height() const -> unsigned
  {
    return m_height;
  }

  inline auto image::depth() const -> unsigned
  {
    return m_depth;
  }

  inline auto image::widthf() const -> float
  {
    return m_widthf;
  }

  inline auto image::heightf() const -> float
  {
    return m_heightf;
  }

  inline auto image::depthf() const -> float
  {
    return m_depthf;
  }


  inline image::image()
      : m_width(0)
      , m_height(0)
      , m_depth(0)
      , m_widthf(0)
      , m_heightf(0)
      , m_depthf(0)
  {
  }

  inline image::image(unsigned width, unsigned height, unsigned depth)
      : m_width(width)
      , m_height(height)
      , m_depth(depth)
      , m_widthf(static_cast<float>(width))
      , m_heightf(static_cast<float>(height))
      , m_depthf(static_cast<float>(depth))
  {
  }

  inline image::image(const image& rhs) = default;

  inline auto image::operator=(const image& rhs) -> image& = default;

  inline image::image(image&& rhs) noexcept
      : m_width(rhs.m_width)
      , m_height(rhs.m_height)
      , m_depth(rhs.m_depth)
      , m_widthf(rhs.m_widthf)
      , m_heightf(rhs.m_heightf)
      , m_depthf(rhs.m_depthf)
  {
  }

  inline auto image::operator=(image&& rhs) noexcept -> image&
  {
    m_width   = rhs.m_width;
    m_height  = rhs.m_height;
    m_depth   = rhs.m_depth;
    m_widthf  = rhs.m_widthf;
    m_heightf = rhs.m_heightf;
    m_depthf  = rhs.m_depthf;
    return *this;
  }

  inline image::~image() noexcept = default;

  template <image_format Format>
  inline image2d<Format>::image2d()
      : image()
      , m_data(nullptr)
      , m_owns_data(false)
  {
  }

  template <image_format Format>
  inline image2d<Format>::image2d(unsigned width, unsigned height)
      : image(width, height, 1U)
      , m_data(new Format[static_cast<size_t>(width) * static_cast<size_t>(height)])
      , m_owns_data(true)
  {
  }

  template <image_format Format>
  inline image2d<Format>::image2d(unsigned width, unsigned height, Format* foreign_data)
      : image(width, height, 1U)
      , m_data(foreign_data)
      , m_owns_data(false)
  {
  }

  template <image_format Format>
  inline image2d<Format>::image2d(image2d&& rhs) noexcept
      : image(rhs)
      , m_data(rhs.m_data)
      , m_owns_data(rhs.m_owns_data)
  {
    rhs.m_data      = nullptr;
    rhs.m_owns_data = false;
  }

  template <image_format Format>
  inline auto image2d<Format>::operator=(image2d&& rhs) noexcept -> image2d<Format>&
  {
    if (std::addressof(rhs) != this)
    {
      image::operator=(rhs);
      this->destroy();
      m_data          = rhs.m_data;
      m_owns_data     = rhs.m_owns_data;
      rhs.m_data      = nullptr;
      rhs.m_owns_data = false;
    }
    return *this;
  }

  template <image_format Format>
  inline image2d<Format>::~image2d() noexcept
  {
    this->destroy();
  }

  template <image_format Format>
  inline void image2d<Format>::clear(const Format& value)
  {
    std::fill(m_data, m_data + this->pixel_count(), value);
  }

  template <image_format Format>
  inline auto image2d<Format>::get(unsigned x, unsigned y) const -> const Format&
  {
    const auto index = details::tex_index_2d(x, y, this->width());
    assert(index < this->pixel_count());
    return m_data[index];
  }

  template <image_format Format>
  inline void image2d<Format>::set(unsigned x, unsigned y, const Format& value)
  {
    const auto index = details::tex_index_2d(x, y, this->width());
    assert(index < this->pixel_count());
    m_data[index] = value;
  }

  template <image_format Format>
  inline auto image2d<Format>::operator[](unsigned index) -> Format&
  {
    assert(index < this->pixel_count());
    return m_data[index];
  }

  template <image_format Format>
  auto image2d<Format>::operator[](unsigned index) const -> const Format&
  {
    assert(index < this->pixel_count());
    return m_data[index];
  }

  template <image_format Format>
  inline auto image2d<Format>::data() -> Format*
  {
    return m_data;
  }

  template <image_format Format>
  inline auto image2d<Format>::data() const -> const Format*
  {
    return m_data;
  }

  template <image_format Format>
  inline auto image2d<Format>::data_span() -> std::span<Format>
  {
    return std::span<Format>(this->data(), this->pixel_count());
  }

  template <image_format Format>
  inline auto image2d<Format>::data_span() const -> std::span<const Format>
  {
    return std::span<Format>(this->data(), this->pixel_count());
  }

  template <image_format Format>
  inline auto image2d<Format>::pixel_count() const -> unsigned
  {
    return this->width() * this->height();
  }

  template <image_format Format>
  inline auto image2d<Format>::size_in_bytes() const -> size_t
  {
    return sizeof(Format) * this->pixel_count();
  }

  template <image_format Format>
  inline auto image2d<Format>::row_pitch() const -> size_t
  {
    return sizeof(Format) * this->width();
  }

  template <image_format Format>
  inline auto image2d<Format>::aspect_ratio() const -> float
  {
    return this->widthf() / this->heightf();
  }

  template <image_format Format>
  inline auto image2d<Format>::owns_data() const -> bool
  {
    return m_owns_data;
  }

  template <image_format Format>
  inline void image2d<Format>::destroy()
  {
    if (m_owns_data && m_data != nullptr)
    {
      delete[] m_data;
      m_data = nullptr;
    }
  }

  namespace details
  {
    template <out_vertex OutVertex>
    inline edge_equation::edge_equation(const OutVertex& v0, const OutVertex& v1)
        : a(v0.sv_pos.y - v1.sv_pos.y)
        , b(v1.sv_pos.x - v0.sv_pos.x)
        , c(-(a * (v0.sv_pos.x + v1.sv_pos.x) + b * (v0.sv_pos.y + v1.sv_pos.y)) / 2)
        , tie(a != 0 ? a > 0 : b > 0)
    {
    }

    inline auto edge_equation::evaluate(float x, float y) const -> float
    {
      return a * x + b * y + c;
    }

    inline auto edge_equation::test(float x, float y) const -> float
    {
      return test(evaluate(x, y));
    }

    inline auto edge_equation::test(float xy) const -> float
    {
      return xy > 0.0F || (xy == 0.0F && tie);
    }

    inline auto edge_equation::step_x(float v, float step) const -> float
    {
      return v + a * step;
    }

    inline auto edge_equation::step_y(float v, float step) const -> float
    {
      return v + b * step;
    }

    inline param_equation::param_equation()
        : a(0.0F)
        , b(0.0F)
        , c(0.0F)
    {
    }

    inline param_equation::param_equation(float                p0,
                                          float                p1,
                                          float                p2,
                                          const edge_equation& e0,
                                          const edge_equation& e1,
                                          const edge_equation& e2,
                                          float                factor)
        : a(factor * (p0 * e0.a + p1 * e1.a + p2 * e2.a))
        , b(factor * (p0 * e0.b + p1 * e1.b + p2 * e2.b))
        , c(factor * (p0 * e0.c + p1 * e1.c + p2 * e2.c))
    {
    }

    inline auto param_equation::evaluate(float x, float y) const -> float
    {
      return a * x + b * y + c;
    }

    inline auto param_equation::step_x(float v, float step) const -> float
    {
      return v + a * step;
    }

    inline auto param_equation::step_y(float v, float step) const -> float
    {
      return v + b * step;
    }

    template <out_vertex T>
    inline triangle_equations<T>::triangle_equations(const T& v0, const T& v1, const T& v2)
        : e0(v1, v2)
        , e1(v2, v0)
        , e2(v0, v1)
        , area2(e0.c + e1.c + e2.c)
    {
      // Cull backfacing triangles.
      if (area2 <= 0.0F)
      {
        z       = {};
        inv_w   = {};
        attribs = {};
        return;
      }
      else
      {
        const auto pos0 = v0.sv_pos;
        const auto pos1 = v1.sv_pos;
        const auto pos2 = v2.sv_pos;

        const auto factor = 1.0F / area2;
        this->z           = param_equation(pos0.z, pos1.z, pos2.z, e0, e1, e2, factor);

        const auto inv_w0 = 1.0F / pos0.w;
        const auto inv_w1 = 1.0F / pos1.w;
        const auto inv_w2 = 1.0F / pos2.w;

        this->inv_w = param_equation(inv_w0, inv_w1, inv_w2, e0, e1, e2, factor);

        using traits = out_vertex_traits<T>;

        const auto v0_ptr = traits::attrib_ptr(v0);
        const auto v1_ptr = traits::attrib_ptr(v1);
        const auto v2_ptr = traits::attrib_ptr(v2);

        for (size_t i = 0; i < traits::num_attribs; ++i)
        {
          attribs[i] = param_equation(v0_ptr[i] * inv_w0, v1_ptr[i] * inv_w1, v2_ptr[i] * inv_w2, e0, e1, e2, factor);
        }
      }
    }

    template <index_type I, size_t Size>
    inline vertex_cache<I, Size>::vertex_cache()
    {
      std::fill(m_input_index.begin(), m_input_index.end(), no_index<I>);
      std::fill(m_output_index.begin(), m_output_index.end(), static_cast<I>(0));
    }

    template <index_type I, size_t Size>
    inline void vertex_cache<I, Size>::clear()
    {
      std::fill(m_input_index.begin(), m_input_index.end(), no_index<I>);
    }

    template <index_type I, size_t Size>
    inline void vertex_cache<I, Size>::set(I in_index, I out_index)
    {
      const auto cache_index      = in_index % Size;
      m_input_index[cache_index]  = in_index;
      m_output_index[cache_index] = out_index;
    }

    template <index_type I, size_t Size>
    inline auto vertex_cache<I, Size>::lookup(I in_index) const -> I
    {
      const auto cache_index = in_index % Size;
      return m_input_index[cache_index] == in_index ? m_output_index[cache_index] : no_index<I>;
    }

    template <out_vertex T>
    inline edge_data::edge_data(const triangle_equations<T>& eqn, float x, float y)
        : ev0(eqn.e0.evaluate(x, y))
        , ev1(eqn.e1.evaluate(x, y))
        , ev2(eqn.e2.evaluate(x, y))
    {
    }

    template <out_vertex T>
    inline void edge_data::step_x(const triangle_equations<T>& eqn, float step)
    {
      ev0 = eqn.e0.step_x(ev0, step);
      ev1 = eqn.e1.step_x(ev1, step);
      ev2 = eqn.e2.step_x(ev2, step);
    }

    template <out_vertex T>
    inline void edge_data::step_y(const triangle_equations<T>& eqn, float step)
    {
      ev0 = eqn.e0.step_y(ev0, step);
      ev1 = eqn.e1.step_y(ev1, step);
      ev2 = eqn.e2.step_y(ev2, step);
    }

    template <out_vertex T>
    inline auto edge_data::test(const triangle_equations<T>& eqn) const -> bool
    {
      return eqn.e0.test(ev0) && eqn.e1.test(ev1) && eqn.e2.test(ev2);
    }

    template <out_vertex OutVertex>
    inline interpolated_vertex<OutVertex>::interpolated_vertex(const triangle_equations<OutVertex>& eqn,
                                                               float                                x,
                                                               float                                y)
        : pos()
        , invW()
        , tmpAttribs()
        , attribs()
    {
      assert(this->tmpAttribs.size() == eqn.attribs.size());
      assert(this->attribs.size() == eqn.attribs.size());

      const auto z    = eqn.z.evaluate(x, y);
      const auto invw = eqn.inv_w.evaluate(x, y);
      const auto w    = 1.0F / invw;

      this->pos  = vec4{x, y, z, w};
      this->invW = invw;

      for (size_t i = 0; i < eqn.attribs.size(); ++i)
      {
        tmpAttribs[i] = eqn.attribs[i].evaluate(x, y);
        attribs[i]    = tmpAttribs[i] * pos.w;
      }
    }

    template <out_vertex OutVertex>
    inline interpolated_vertex<OutVertex>::interpolated_vertex(const OutVertex& v)
        : pos()
        , invW()
        , tmpAttribs()
        , attribs()
    {
      using traits = out_vertex_traits<OutVertex>;

      this->pos  = vec4{v.sv_pos};
      this->invW = 1.0F / this->pos.w;

      constexpr auto attrib_count = traits::num_attribs;
      const auto     attrib_ptr   = traits::attrib_ptr(v);

      for (size_t i = 0; i < attrib_count; ++i)
      {
        this->tmpAttribs[i] = attrib_ptr[i];
        this->attribs[i]    = this->tmpAttribs[i];
      }
    }

    template <out_vertex OutVertex>
    void interpolated_vertex<OutVertex>::step_x(const triangle_equations<OutVertex>& eqn)
    {
      assert(this->tmpAttribs.size() == eqn.attribs.size());
      assert(this->attribs.size() == eqn.attribs.size());

      this->pos.z = eqn.z.step_x(this->pos.z, 1.0F);
      this->invW  = eqn.inv_w.step_x(this->invW, 1.0F);
      this->pos.w = 1.0F / this->invW;

      for (size_t i = 0; i < eqn.attribs.size(); ++i)
      {
        this->tmpAttribs[i] = eqn.attribs[i].step_x(this->tmpAttribs[i], 1.0F);
        this->attribs[i]    = this->tmpAttribs[i] * this->pos.w;
      }
    }

    template <out_vertex OutVertex>
    void interpolated_vertex<OutVertex>::step_y(const triangle_equations<OutVertex>& eqn)
    {
      this->pos.z = eqn.z.step_y(this->pos.z, 1.0F);
      this->invW  = eqn.inv_w.step_x(this->invW, 1.0F);
      this->pos.w = 1.0F / this->invW;

      for (size_t i = 0; i < eqn.attribs.size(); ++i)
      {
        this->tmpAttribs[i] = eqn.attribs[i].step_y(this->tmpAttribs[i], 1.0F);
        this->attribs[i]    = this->tmpAttribs[i] * this->pos.w;
      }
    }

    template <out_vertex OutVertex>
    auto interpolated_vertex<OutVertex>::to_out_vertex() const -> OutVertex
    {
      using traits = out_vertex_traits<OutVertex>;

      OutVertex out_vertex;
      this->pos.assign_to(out_vertex.sv_pos);

      constexpr auto num_attribs = traits::num_attribs;
      auto           dst_ptr     = traits::attrib_ptr(out_vertex);

      for (size_t i = 0; i < num_attribs; ++i, ++dst_ptr)
      {
        *dst_ptr = this->attribs[i];
      }

      return out_vertex;
    }

    template <index_type I>
    inline poly_clipper<I>::poly_clipper(size_t initial_capacity)
    {
      m_indices_in.reserve(initial_capacity);
      m_indices_out.reserve(initial_capacity);
    }

    template <index_type I>
    void poly_clipper<I>::clear()
    {
      m_indices_in.clear();
      m_indices_out.clear();
    }

    template <index_type I>
    void poly_clipper<I>::init(I i1, I i2, I i3)
    {
      m_indices_in.clear();
      m_indices_out.clear();

      m_indices_in.push_back(i1);
      m_indices_in.push_back(i2);
      m_indices_in.push_back(i3);
    }

    template <out_vertex OutVertex>
    static auto lerp_out_vertex(const OutVertex& start, const OutVertex& end, float t) -> OutVertex
    {
      using traits = out_vertex_traits<OutVertex>;

      constexpr auto float_count = traits::num_floats;

      auto result    = OutVertex();
      auto dst_ptr   = traits::float_ptr(result);
      auto start_ptr = traits::float_ptr(start);
      auto end_ptr   = traits::float_ptr(end);

      for (size_t i = 0; i < float_count; ++i, ++dst_ptr, ++start_ptr, ++end_ptr)
      {
        *dst_ptr = std::lerp(*start_ptr, *end_ptr, t);
      }

      return result;
    }

    template <index_type I>
    template <out_vertex T>
    void poly_clipper<I>::clip_to_plane(std::vector<T>& vertices, float a, float b, float c, float d)
    {
      if (this->is_fully_clipped())
      {
        return;
      }

      m_indices_out.clear();

      auto idx_prev = m_indices_in.front();
      m_indices_in.push_back(idx_prev);

      const auto& v_prev     = vertices[idx_prev];
      const auto  v_prev_pos = v_prev.sv_pos;
      auto        dp_prev    = a * v_prev_pos.x + b * v_prev_pos.y + c * v_prev_pos.z + d * v_prev_pos.w;

      for (size_t i = 1; i < m_indices_in.size(); ++i)
      {
        const auto  idx   = m_indices_in[i];
        const auto& v     = vertices[idx];
        const auto  v_pos = v.sv_pos;
        const auto  dp    = a * v_pos.x + b * v_pos.y + c * v_pos.z + d * v_pos.w;

        if (dp_prev >= 0.0F)
        {
          m_indices_out.push_back(idx_prev);
        }

        if (sign(dp) != sign(dp_prev))
        {
          const auto t     = dp < 0.0F ? dp_prev / (dp_prev - dp) : -dp_prev / (dp - dp_prev);
          const auto v_out = lerp_out_vertex(vertices[idx_prev], vertices[idx], t);
          vertices.push_back(v_out);
          m_indices_out.push_back(static_cast<I>(vertices.size() - 1));
        }

        idx_prev = idx;
        dp_prev  = dp;
      }

      std::swap(m_indices_in, m_indices_out);
    }

    template <index_type I>
    auto poly_clipper<I>::indices() const -> const std::vector<I>&
    {
      return m_indices_in;
    }

    template <index_type I>
    auto poly_clipper<I>::is_fully_clipped() const -> bool
    {
      return m_indices_in.size() < 3;
    }

    template <out_vertex T>
    line_clipper::line_clipper(const T& v0, const T& v1)
        : v0(v0.sv_pos)
        , v1(v1.sv_pos)
        , t1(1.0F)
    {
    }

    inline void line_clipper::clip_to_plane(float a, float b, float c, float d)
    {
      if (fully_clipped)
      {
        return;
      }

      const auto dp0 = a * v0.x + b * v0.y + c * v0.z + d * v0.w;
      const auto dp1 = a * v1.x + b * v1.y + c * v1.z + d * v1.w;

      const auto dp0_neg = dp0 < 0.0F;
      const auto dp1_neg = dp1 < 0.0F;

      if (dp0_neg && dp1_neg)
      {
        fully_clipped = true;
        return;
      }

      if (dp0_neg)
      {
        const auto t = -dp0 / (dp1 - dp0);
        t0           = std::max(t0, t);
      }
      else
      {
        const auto t = dp0 / (dp0 - dp1);
        t1           = std::min(t1, t);
      }
    }

    inline auto tex_coord_to_index(float coord, unsigned size, image_address_mode mode) -> unsigned
    {
      const auto coord_int = static_cast<unsigned>(coord * static_cast<float>(size));

      if (coord >= 0.0F && coord <= 1.0F)
      {
        return coord_int - 1;
      }
      else
      {
        switch (mode)
        {
          case image_address_mode::wrap: return coord_int % size;
          case image_address_mode::clamp_to_edge: return std::clamp(coord_int, 0U, size - 1);
          case image_address_mode::mirror: {
            const auto modfn    = [](int a, int n) { return (a % n + n) % n; };
            const auto mirrorfn = [](int a) { return a >= 0 ? a : -(1 + a); };
            const auto repeatfn = [&](int x) { return (size - 1) - mirrorfn(modfn(x, 2 * size) - size); };

            return repeatfn(static_cast<int>(coord_int));
          }
        }
        return 0U;
      }
    }

    inline auto tex_coord_to_index_absolute(float coord, float size, image_address_mode mode) -> float
    {
      if (coord >= 0.0F && coord <= 1.0F)
      {
        return coord;
      }
      else
      {
        switch (mode)
        {
          case image_address_mode::wrap: return std::fmod(coord, size);
          case image_address_mode::clamp_to_edge: return std::clamp(coord, 0.0F, size - 1.0F);
          case image_address_mode::mirror: {
            const auto modfn    = [](float a, float n) { return std::fmod((std::fmod(a, n) + n), n); };
            const auto mirrorfn = [](float a) { return a > 0.0F ? a : -(1.0F + a); };
            const auto repeatfn = [&](float x) { return (size - 1.0F) - mirrorfn(modfn(x, 2.0F * size) - size); };

            return repeatfn(coord);
          }
        }
        return 0U;
      }
    }

    static inline auto primitive_count(size_t index_count, primitive_topology topology) -> size_t
    {
      switch (topology)
      {
        case primitive_topology::point: return index_count;
        case primitive_topology::line: return index_count / 2;
        case primitive_topology::triangle_list: return index_count / 3;
      }
      return 0;
    }

    template <out_vertex OutVertex>
    static auto clip_mask(const OutVertex& vertex) -> uint8_t
    {
      const auto pos = vertex.sv_pos;

      uint8_t mask = 0;

      if (pos.w - pos.x < 0.0F)
      {
        mask |= static_cast<uint8_t>(vertex_clip_mask::pos_x);
      }

      if (pos.x + pos.w < 0.0F)
      {
        mask |= static_cast<uint8_t>(vertex_clip_mask::neg_x);
      }

      if (pos.w - pos.y < 0.0F)
      {
        mask |= static_cast<uint8_t>(vertex_clip_mask::pos_y);
      }

      if (pos.y + pos.w < 0.0F)
      {
        mask |= static_cast<uint8_t>(vertex_clip_mask::neg_y);
      }

      if (pos.w - pos.z < 0.0F)
      {
        mask |= static_cast<uint8_t>(vertex_clip_mask::pos_z);
      }

      if (pos.z + pos.w < 0.0F)
      {
        mask |= static_cast<uint8_t>(vertex_clip_mask::neg_z);
      }

      return mask;
    }

    template <out_vertex OutVertex, index_type IndexType, size_t VertexCacheSize>
    static inline void clip_points(raster_cache<OutVertex, IndexType, VertexCacheSize>& cache)
    {
      cache.clip_mask.clear();

      for (size_t i = 0; i < cache.vertices_out.size(); ++i)
      {
        cache.clip_mask.push_back(clip_mask(cache.vertices_out[i]));
      }

      for (size_t i = 0; i < cache.indices_out.size(); ++i)
      {
        if (cache.clip_mask[cache.indices_out[i]] != 0)
        {
          cache.indices_out[i] = no_index<IndexType>;
        }
      }
    }

    template <out_vertex OutVertex, index_type IndexType, size_t VertexCacheSize>
    static void clip_lines(raster_cache<OutVertex, IndexType, VertexCacheSize>& cache)
    {
      cache.clip_mask.clear();

      for (size_t i = 0; i < cache.vertices_out.size(); ++i)
      {
        cache.clip_mask.push_back(clip_mask(cache.vertices_out[i]));
      }

      size_t i = 0;
      while ((i + 2) < cache.indices_out.size())
      {
        const auto index0 = static_cast<size_t>(cache.indices_out[i]);
        const auto index1 = static_cast<size_t>(cache.indices_out[i + 1]);

        const auto& v0 = cache.vertices_out[index0];
        const auto& v1 = cache.vertices_out[index1];

        const auto mask = static_cast<vertex_clip_mask>(cache.clip_mask[index0] | cache.clip_mask[index1]);

        auto line_clipper = details::line_clipper(v0, v1);

        if (mask & vertex_clip_mask::pos_x)
        {
          line_clipper.clip_to_plane(-1.0F, 0.0F, 0.0F, 1.0F);
        }

        if (mask & vertex_clip_mask::neg_x)
        {
          line_clipper.clip_to_plane(1.0F, 0.0F, 0.0F, 1.0F);
        }

        if (mask & vertex_clip_mask::pos_y)
        {
          line_clipper.clip_to_plane(0.0F, -1.0F, 0.0F, 1.0F);
        }

        if (mask & vertex_clip_mask::neg_y)
        {
          line_clipper.clip_to_plane(0.0F, 1.0F, 0.0F, 1.0F);
        }

        if (mask & vertex_clip_mask::pos_z)
        {
          line_clipper.clip_to_plane(0.0F, 0.0F, -1.0F, 1.0F);
        }

        if (mask & vertex_clip_mask::neg_z)
        {
          line_clipper.clip_to_plane(0.0F, 0.0F, 1.0F, 1.0F);
        }

        if (line_clipper.fully_clipped)
        {
          cache.indices_out[i]     = no_index<IndexType>;
          cache.indices_out[i + 1] = no_index<IndexType>;
          i += 2;
          continue;
        }

        if (cache.clip_mask[index0] != 0)
        {
          const auto new_v = lerp_out_vertex(v0, v1, line_clipper.t0);
          cache.vertices_out.push_back(new_v);
          cache.indices_out[i] = static_cast<IndexType>(cache.vertices_out.size()) - 1;
        }

        if (cache.clip_mask[index1] != 0)
        {
          const auto new_v = lerp_out_vertex(v0, v1, line_clipper.t1);
          cache.vertices_out.push_back(new_v);
          cache.indices_out[i + 1] = static_cast<IndexType>(cache.vertices_out.size()) - 1;
        }

        i += 2;
      }
    }

    template <out_vertex OutVertex, index_type IndexType, size_t VertexCacheSize>
    static void clip_triangles(raster_cache<OutVertex, IndexType, VertexCacheSize>& cache)
    {
      cache.clip_mask.clear();

      for (size_t i = 0; i < cache.vertices_out.size(); ++i)
      {
        cache.clip_mask.push_back(clip_mask(cache.vertices_out[i]));
      }

      const auto n = cache.indices_out.size();

      for (size_t i = 0; (i + 2) < n; i += 3)
      {
        const auto i0 = cache.indices_out[i];
        const auto i1 = cache.indices_out[i + 1];
        const auto i2 = cache.indices_out[i + 2];

        const auto clip_mask =
            static_cast<vertex_clip_mask>(cache.clip_mask[i0] | cache.clip_mask[i1] | cache.clip_mask[i2]);

        cache.pclipper.init(i0, i1, i2);

        if (clip_mask & vertex_clip_mask::pos_x)
          cache.pclipper.clip_to_plane(cache.vertices_out, -1, 0, 0, 1);
        if (clip_mask & vertex_clip_mask::neg_x)
          cache.pclipper.clip_to_plane(cache.vertices_out, 1, 0, 0, 1);
        if (clip_mask & vertex_clip_mask::pos_y)
          cache.pclipper.clip_to_plane(cache.vertices_out, 0, -1, 0, 1);
        if (clip_mask & vertex_clip_mask::neg_y)
          cache.pclipper.clip_to_plane(cache.vertices_out, 0, 1, 0, 1);
        if (clip_mask & vertex_clip_mask::pos_z)
          cache.pclipper.clip_to_plane(cache.vertices_out, 0, 0, -1, 1);
        if (clip_mask & vertex_clip_mask::neg_z)
          cache.pclipper.clip_to_plane(cache.vertices_out, 0, 0, 1, 1);

        if (cache.pclipper.is_fully_clipped())
        {
          cache.indices_out[i]     = no_index<IndexType>;
          cache.indices_out[i + 1] = no_index<IndexType>;
          cache.indices_out[i + 2] = no_index<IndexType>;
          continue;
        }

        const auto& indices = cache.pclipper.indices();

        cache.indices_out[i]     = indices[0];
        cache.indices_out[i + 1] = indices[1];
        cache.indices_out[i + 2] = indices[2];

        for (size_t idx = 3; idx < indices.size(); ++idx)
        {
          cache.indices_out.push_back(indices[0]);
          cache.indices_out.push_back(indices[idx - 1]);
          cache.indices_out.push_back(indices[idx]);
        }
      }
    }

    template <out_vertex OutVertex, index_type IndexType, size_t VertexCacheSize>
    static void clip_primitives(raster_cache<OutVertex, IndexType, VertexCacheSize>& cache, primitive_topology topology)
    {
      switch (topology)
      {
        case primitive_topology::point: clip_points(cache); break;
        case primitive_topology::line: clip_lines(cache); break;
        case primitive_topology::triangle_list: clip_triangles(cache); break;
      }
    }

    static inline auto perspective_divide(const vec4& v) -> vec4
    {
      const auto inv_w = 1.0F / v.w;

      return {
          v.x * inv_w,
          v.y * inv_w,
          v.z * inv_w,
          v.w,
      };
    }

    static inline auto clip_space_to_screen_space(
        const vec4& v, float vpx, float vpy, float vpw, float vph, float vpzmin, float vpzmax) -> vec4
    {
      return {
          (v.x + 1.0F) * vpw * 0.5F + vpx,
          (1.0F - v.y) * vph * 0.5F + vpy,
          vpzmin + v.z * (vpzmax - vpzmin),
          v.w,
      };
    }

    template <out_vertex OutVertex, index_type IndexType, size_t VertexCacheSize>
    static void transform_vertices(const viewport& viewport, raster_cache<OutVertex, IndexType, VertexCacheSize>& cache)
    {
      cache.already_processed.clear();
      for (size_t i = 0; i < cache.vertices_out.size(); ++i)
      {
        cache.already_processed.push_back(false);
      }

      const auto vpx    = static_cast<float>(viewport.rect.x);
      const auto vpy    = static_cast<float>(viewport.rect.y);
      const auto vpw    = static_cast<float>(viewport.rect.width);
      const auto vph    = static_cast<float>(viewport.rect.height);
      const auto vpzmin = viewport.zmin;
      const auto vpzmax = viewport.zmax;

      for (const IndexType index : cache.indices_out)
      {
        if (index == no_index<IndexType>)
        {
          continue;
        }

        if (cache.already_processed[index])
        {
          continue;
        }

        auto& out_vertex = cache.vertices_out[index];

        auto new_pos = vec4(out_vertex.sv_pos);
        new_pos      = perspective_divide(new_pos);
        new_pos      = clip_space_to_screen_space(new_pos, vpx, vpy, vpw, vph, vpzmin, vpzmax);
        new_pos.assign_to(out_vertex.sv_pos);

        cache.already_processed[index] = true;
      }
    }

    static inline auto scissor_test(const rect& rect, int x, int y) -> bool
    {
      return x >= rect.x && x < rect.x + static_cast<int>(rect.width) && y >= rect.y &&
             y < rect.y + static_cast<int>(rect.height);
    }

    template <depth_format DepthFormat>
    static auto depth_test(unsigned                    x,
                           unsigned                    y,
                           float                       z,
                           const depth_stencil_state&  depth_stencil_state,
                           const image2d<DepthFormat>& depth_target) -> bool
    {
      if (!depth_stencil_state.is_depth_testing_enabled)
      {
        return true;
      }

      if constexpr (std::is_same_v<DepthFormat, no_depth>)
      {
        return true;
      }
      else
      {
        const auto stored_depth = read_image_depth(depth_target, x, y);

        switch (depth_stencil_state.depth_test_func)
        {
          case compare_func::always: return true;
          case compare_func::never: return false;
          case compare_func::less: return z < stored_depth;
          case compare_func::less_equal: return z <= stored_depth;
          case compare_func::equal: return z == stored_depth;
          case compare_func::greater_equal: return z >= stored_depth;
          case compare_func::greater: return z > stored_depth;
          case compare_func::not_equal: return z != stored_depth;
        }

        return true;
      }
    }

    static inline auto do_alpha_blend(const rgba32f& src, const rgba32f& dst, const blend_state& blend_state) -> rgba32f
    {
      // https://docs.microsoft.com/en-us/windows/win32/api/d3d11/ne-d3d11-d3d11_blend
      const auto modulate_color = [&](const rgba32f& color, blend blend) -> rgb32f {
        return color.to_rgb() * [&]() -> rgb32f {
          switch (blend)
          {
            case blend::one: return {1.0F, 1.0F, 1.0F};
            case blend::zero: return {0.0F, 0.0F, 0.0F};
            case blend::src_color: return src.to_rgb();
            case blend::inverse_src_color: return {1.0F - src.r, 1.0F - src.g, 1.0F - src.b};
            case blend::src_alpha: return rgb32f{src.a, src.a, src.a};
            case blend::inverse_src_alpha: {
              const float comp = 1.0F - src.a;
              return rgb32f{comp, comp, comp};
            }
            case blend::dst_color: return dst.to_rgb();
            case blend::inverse_dst_color: return {1.0F - dst.r, 1.0F - dst.g, 1.0F - dst.b};
            case blend::dst_alpha: return rgb32f{dst.a, dst.a, dst.a};
            case blend::inv_dst_alpha: {
              const float comp = 1.0F - dst.a;
              return rgb32f{comp, comp, comp};
            }
            case blend::blend_factor: return blend_state.blend_factor.to_rgb();
            case blend::inverse_blend_factor:
              return {
                  1.0F - blend_state.blend_factor.r,
                  1.0F - blend_state.blend_factor.g,
                  1.0F - blend_state.blend_factor.b,
              };
            case blend::src_alpha_saturation: {
              const float comp = std::min(src.a, 1.0F - dst.a);
              return rgb32f{comp, comp, comp};
            }
          }

          return rgb32f{};
        }();
      };

      const auto final_rgb = [&]() -> rgb32f {
        const auto color1 = modulate_color(src, blend_state.src_blend_rgb);
        const auto color2 = modulate_color(dst, blend_state.dst_blend_rgb);

        switch (blend_state.blend_op_rgb)
        {
          case blend_op::add:
            return {
                color1.r + color2.r,
                color1.g + color2.g,
                color1.b + color2.b,
            };
          case blend_op::subtract:
            return {
                color1.r - color2.r,
                color1.g - color2.g,
                color1.b - color2.b,
            };
          case blend_op::reverse_subtract:
            return {
                color2.r - color1.r,
                color2.g - color1.g,
                color2.b - color1.b,
            };
          case blend_op::min:
            return rgb32f{
                std::min(color1.r, color2.r),
                std::min(color1.g, color2.g),
                std::min(color1.b, color2.b),
            };
          case blend_op::max:
            return rgb32f{
                std::max(color1.r, color2.r),
                std::max(color1.g, color2.g),
                std::max(color1.b, color2.b),
            };
        }

        return rgb32f{};
      }();

      const auto modulate_alpha = [&](float alpha, blend blend) -> float {
        return alpha * [&]() -> float {
          switch (blend)
          {
            case blend::one: return 1.0F;
            case blend::zero: return 0.0F;
            case blend::src_color: return src.a;
            case blend::inverse_src_color: return 1.0F - src.a;
            case blend::src_alpha: return src.a;
            case blend::inverse_src_alpha: return 1.0F - src.a;
            case blend::dst_color: return dst.a;
            case blend::inverse_dst_color: return 1.0F - dst.a;
            case blend::dst_alpha: return dst.a;
            case blend::inv_dst_alpha: return 1.0F - dst.a;
            case blend::blend_factor: return blend_state.blend_factor.a;
            case blend::inverse_blend_factor: return 1.0F - blend_state.blend_factor.a;
            case blend::src_alpha_saturation: return 1.0F;
          }

          return 0.0F;
        }();
      };

      const auto final_alpha = [&]() -> float {
        const auto alpha1 = modulate_alpha(src.a, blend_state.src_blend_alpha);
        const auto alpha2 = modulate_alpha(dst.a, blend_state.dst_blend_alpha);

        switch (blend_state.blend_op_rgb)
        {
          case blend_op::add: return alpha1 + alpha2;
          case blend_op::subtract: return alpha1 - alpha2;
          case blend_op::reverse_subtract: return alpha2 - alpha1;
          case blend_op::min: return std::min(alpha1, alpha2);
          case blend_op::max: return std::max(alpha1, alpha2);
        }

        return 0.0F;
      }();

      return rgba32f(final_rgb, final_alpha);
    }

    template <color_format ColorFormat>
    static void write_pixel(image2d<ColorFormat>& color_target,
                            const rgba32f&        src_color,
                            const blend_state&    blend_state,
                            unsigned              x,
                            unsigned              y)
    {
      const auto blended_color = [&] {
        if (blend_state.is_blending_enabled)
        {
          const auto dst_color = read_image_color(color_target, x, y);
          return do_alpha_blend(src_color, dst_color, blend_state);
        }
        else
        {
          return src_color;
        }
      }();

      //.clamp(&rgba32f::zero(), &rgba32f::one());

      put_image_color(color_target, x, y, blended_color);
    }

    template <size_t SrcColorCount, size_t I = 0, typename... Ts>
    constexpr void write_pixel_tuple(std::tuple<Ts...>&                        images_tuple,
                                     const std::array<rgba32f, SrcColorCount>& src_colors,
                                     const blend_state&                        blend_state,
                                     unsigned                                  x,
                                     unsigned                                  y)
    {
      if constexpr (I != sizeof...(Ts))
      {
        auto& color_target = std::get<I>(images_tuple);
        details::write_pixel(color_target, src_colors[I], blend_state, x, y);
        write_pixel_tuple<SrcColorCount, I + 1>(images_tuple, src_colors, blend_state, x, y);
      }
    }

    template <out_vertex OutVertex,
              typename ColorImages,
              depth_format                                DepthFormat,
              pixel_shader<OutVertex, ColorImages::count> PixelShader>
    static void draw_pixel(const rect&                           scissor_rect,
                           const PixelShader&                    pixel_shader,
                           ColorImages&                          color_targets,
                           image2d<DepthFormat>&                 depth_target,
                           const blend_state&                    blend_state,
                           const depth_stencil_state&            depth_stencil_state,
                           const interpolated_vertex<OutVertex>& p)
    {
      const auto xi = static_cast<int>(p.pos.x);
      const auto yi = static_cast<int>(p.pos.y);

      if (!scissor_test(scissor_rect, xi, yi))
      {
        return;
      }

      const auto x = static_cast<unsigned>(xi);
      const auto y = static_cast<unsigned>(yi);

      const auto out_vertex = p.to_out_vertex();
      const auto depth      = out_vertex.sv_pos.z;

      if (!depth_test(x, y, depth, depth_stencil_state, depth_target))
      {
        return;
      }

      const auto src_colors = pixel_shader.execute(out_vertex);

      details::write_pixel_tuple(color_targets.images(), src_colors, blend_state, x, y);

      if (depth_stencil_state.is_depth_write_enabled)
      {
        if constexpr (!std::is_same_v<DepthFormat, no_depth>)
        {
          put_image_depth(depth_target, x, y, depth);
        }
      }
    }

    template <out_vertex OutVertex,
              typename ColorImages,
              depth_format                                DepthFormat,
              pixel_shader<OutVertex, ColorImages::count> PixelShader>
    static void draw_point(const rect&                scissor_rect,
                           const PixelShader&         pixel_shader,
                           ColorImages&               color_targets,
                           image2d<DepthFormat>&      depth_target,
                           const blend_state&         blend_state,
                           const depth_stencil_state& depth_stencil_state,
                           const OutVertex&           v)
    {
      const interpolated_vertex<OutVertex> p{v};
      draw_pixel(scissor_rect, pixel_shader, color_targets, depth_target, blend_state, depth_stencil_state, p);
    }

    template <out_vertex OutVertex,
              index_type IndexType,
              typename ColorImages,
              depth_format                                DepthFormat,
              pixel_shader<OutVertex, ColorImages::count> PixelShader>
    static void draw_point_list(const rect&                   scissor_rect,
                                const PixelShader&            pixel_shader,
                                ColorImages&                  color_targets,
                                image2d<DepthFormat>&         depth_target,
                                const blend_state&            blend_state,
                                const depth_stencil_state&    depth_stencil_state,
                                const std::vector<OutVertex>& vertices,
                                const std::vector<IndexType>& indices)
    {
      for (const auto& idx : indices)
      {
        if (idx == no_index<IndexType>)
        {
          continue;
        }

        draw_point(scissor_rect,
                   pixel_shader,
                   color_targets,
                   depth_target,
                   blend_state,
                   depth_stencil_state,
                   vertices[idx]);
      }
    }

    template <out_vertex OutVertex>
    static auto add_out_vertex(const OutVertex& lhs, const OutVertex& rhs) -> OutVertex
    {
      using traits = out_vertex_traits<OutVertex>;

      auto result  = OutVertex();
      auto dst_ptr = traits::float_ptr(result);
      auto lhs_ptr = traits::float_ptr(lhs);
      auto rhs_ptr = traits::float_ptr(rhs);

      constexpr auto num_floats = traits::num_floats;

      for (size_t i = 0; i < num_floats; ++i, ++dst_ptr, ++lhs_ptr, ++rhs_ptr)
      {
        *dst_ptr = *lhs_ptr + *rhs_ptr;
      }

      return result;
    }

    template <out_vertex OutVertex>
    static auto sub_out_vertex(const OutVertex& lhs, const OutVertex& rhs) -> OutVertex
    {
      using traits = out_vertex_traits<OutVertex>;

      auto result  = OutVertex();
      auto dst_ptr = traits::float_ptr(result);
      auto lhs_ptr = traits::float_ptr(lhs);
      auto rhs_ptr = traits::float_ptr(rhs);

      constexpr auto num_floats = traits::num_floats;

      for (size_t i = 0; i < num_floats; ++i, ++dst_ptr, ++lhs_ptr, ++rhs_ptr)
      {
        *dst_ptr = *lhs_ptr - *rhs_ptr;
      }

      return result;
    }

    template <out_vertex OutVertex>
    static auto divf_out_vertex(const OutVertex& lhs, float rhs) -> OutVertex
    {
      using traits = out_vertex_traits<OutVertex>;

      OutVertex result;
      auto      dst_ptr = traits::float_ptr(result);
      auto      lhs_ptr = traits::float_ptr(lhs);

      constexpr auto num_floats = traits::num_floats;

      for (size_t i = 0; i < num_floats; ++i, ++dst_ptr, ++lhs_ptr)
      {
        *dst_ptr = *lhs_ptr / rhs;
      }

      return result;
    }

    template <out_vertex OutVertex>
    static auto compute_vertex_step(const OutVertex& v0, const OutVertex& v1, float steps) -> OutVertex
    {
      return divf_out_vertex(sub_out_vertex(v1, v0), steps);
    }

    template <out_vertex OutVertex,
              typename ColorImages,
              depth_format                                DepthFormat,
              pixel_shader<OutVertex, ColorImages::count> PixelShader>
    static void draw_line(const rect&                scissor_rect,
                          const PixelShader&         pixel_shader,
                          ColorImages&               color_targets,
                          image2d<DepthFormat>&      depth_target,
                          const blend_state&         blend_state,
                          const depth_stencil_state& depth_stencil_state,
                          const OutVertex&           v0,
                          const OutVertex&           v1)
    {
      const vec4 pos0{v0.sv_pos};
      const vec4 pos1{v1.sv_pos};

      const float adx = std::abs(pos1.x - pos0.x);
      const float ady = std::abs(pos1.y - pos0.y);

      float steps = std::max(adx, ady);

      const auto step = compute_vertex_step(v0, v1, steps);
      OutVertex  v    = v0;

      steps -= 2.0F;
      while (steps > 0.0F)
      {
        const interpolated_vertex<OutVertex> p{v};
        draw_pixel(scissor_rect, pixel_shader, color_targets, depth_target, blend_state, depth_stencil_state, p);
        v = add_out_vertex(v, step);
        steps -= 1.0F;
      }
    }

    template <out_vertex OutVertex,
              index_type IndexType,
              typename ColorImages,
              depth_format                                DepthFormat,
              pixel_shader<OutVertex, ColorImages::count> PixelShader>
    static void draw_line_list(const rect&                   scissor_rect,
                               const PixelShader&            pixel_shader,
                               ColorImages&                  color_targets,
                               image2d<DepthFormat>&         depth_target,
                               const blend_state&            blend_state,
                               const depth_stencil_state&    depth_stencil_state,
                               const std::vector<OutVertex>& vertices,
                               const std::vector<IndexType>& indices)
    {
      size_t i = 0;
      while (i + 2 <= indices.size())
      {
        if (indices[i] == no_index<IndexType>)
        {
          i += 2;
          continue;
        }

        const auto& v0 = vertices[indices[i]];
        const auto& v1 = vertices[indices[i + 1]];
        draw_line(scissor_rect, pixel_shader, color_targets, depth_target, blend_state, depth_stencil_state, v0, v1);
        i += 2;
      }
    }


    template <out_vertex OutVertex,
              typename ColorImages,
              depth_format                                DepthFormat,
              pixel_shader<OutVertex, ColorImages::count> PixelShader>
    static void draw_pixel_span(const rect&                          scissor_rect,
                                const PixelShader&                   pixel_shader,
                                ColorImages&                         color_targets,
                                image2d<DepthFormat>&                depth_target,
                                const blend_state&                   blend_state,
                                const depth_stencil_state&           depth_stencil_state,
                                const triangle_equations<OutVertex>& eqn,
                                int                                  x,
                                int                                  y,
                                int                                  x2)
    {
      const auto xf = static_cast<float>(x) + 0.5F;
      const auto yf = static_cast<float>(y) + 0.5F;

      interpolated_vertex<OutVertex> p{eqn, xf, yf};

      while (x < x2)
      {
        p.pos.x = static_cast<float>(x);
        draw_pixel(scissor_rect, pixel_shader, color_targets, depth_target, blend_state, depth_stencil_state, p);
        p.step_x(eqn);
        ++x;
      }
    }


    template <out_vertex OutVertex,
              typename ColorImages,
              depth_format                                DepthFormat,
              pixel_shader<OutVertex, ColorImages::count> PixelShader>
    static void draw_top_flat_triangle(const rect&                          scissor_rect,
                                       const PixelShader&                   pixel_shader,
                                       ColorImages&                         color_targets,
                                       image2d<DepthFormat>&                depth_target,
                                       const blend_state&                   blend_state,
                                       const depth_stencil_state&           depth_stencil_state,
                                       const triangle_equations<OutVertex>& eqn,
                                       const OutVertex&                     v0,
                                       const OutVertex&                     v1,
                                       const OutVertex&                     v2)
    {
      const auto v0pos = v0.sv_pos;
      const auto v1pos = v1.sv_pos;
      const auto v2pos = v2.sv_pos;

      const auto invslope1 = (v2pos.x - v0pos.x) / (v2pos.y - v0pos.y);
      const auto invslope2 = (v2pos.x - v1pos.x) / (v2pos.y - v1pos.y);
      const auto from      = static_cast<int>(v2pos.y - 0.5F);
      const auto to        = static_cast<int>(v0pos.y - 0.5F);

      const auto scissor_min_x = scissor_rect.x;
      const auto scissor_max_x = scissor_rect.x + static_cast<int>(scissor_rect.width);

      for (int scanline_y = from; scanline_y > to; --scanline_y)
      {
        const auto dy    = static_cast<float>(scanline_y) - v2pos.y + 0.5F;
        const auto curx1 = v2pos.x + invslope1 * dy + 0.5F;
        const auto curx2 = v2pos.x + invslope2 * dy + 0.5F;

        // Clip to scissor rect
        const auto xl = std::max(scissor_min_x, static_cast<int>(curx1));
        const auto xr = std::min(scissor_max_x, static_cast<int>(curx2));

        draw_pixel_span(scissor_rect,
                        pixel_shader,
                        color_targets,
                        depth_target,
                        blend_state,
                        depth_stencil_state,
                        eqn,
                        xl,
                        scanline_y,
                        xr);
      }
    }

    template <out_vertex OutVertex,
              typename ColorImages,
              depth_format                                DepthFormat,
              pixel_shader<OutVertex, ColorImages::count> PixelShader>
    static void draw_bottom_flat_triangle(const rect&                          scissor_rect,
                                          const PixelShader&                   pixel_shader,
                                          ColorImages&                         color_targets,
                                          image2d<DepthFormat>&                depth_target,
                                          const blend_state&                   blend_state,
                                          const depth_stencil_state&           depth_stencil_state,
                                          const triangle_equations<OutVertex>& eqn,
                                          const OutVertex&                     v0,
                                          const OutVertex&                     v1,
                                          const OutVertex&                     v2)
    {
      const auto v0pos = v0.sv_pos;
      const auto v1pos = v1.sv_pos;
      const auto v2pos = v2.sv_pos;

      const auto invslope1 = (v1pos.x - v0pos.x) / (v1pos.y - v0pos.y);
      const auto invslope2 = (v2pos.x - v0pos.x) / (v2pos.y - v0pos.y);
      const auto from      = static_cast<int>(v0pos.y + 0.5F);
      const auto to        = static_cast<int>(v1pos.y + 0.5F);

      const auto scissor_min_x = static_cast<int>(scissor_rect.x);
      const auto scissor_max_x = static_cast<int>(scissor_rect.x) + static_cast<int>(scissor_rect.width);

      for (int scanline_y = from; scanline_y < to; ++scanline_y)
      {
        const auto dy    = static_cast<float>(scanline_y) - v0pos.y + 0.5F;
        const auto curx1 = v0pos.x + invslope1 * dy + 0.5F;
        const auto curx2 = v0pos.x + invslope2 * dy + 0.5F;

        // Clip to scissor rect
        const auto xl = std::max(scissor_min_x, static_cast<int>(curx1));
        const auto xr = std::min(scissor_max_x, static_cast<int>(curx2));

        draw_pixel_span(scissor_rect,
                        pixel_shader,
                        color_targets,
                        depth_target,
                        blend_state,
                        depth_stencil_state,
                        eqn,
                        xl,
                        scanline_y,
                        xr);
      }
    }

    template <out_vertex OutVertex,
              typename ColorImages,
              depth_format                                DepthFormat,
              pixel_shader<OutVertex, ColorImages::count> PixelShader>
    static void draw_triangle_spanwise(const rect&                scissor_rect,
                                       const PixelShader&         pixel_shader,
                                       ColorImages&               color_targets,
                                       image2d<DepthFormat>&      depth_target,
                                       const blend_state&         blend_state,
                                       const depth_stencil_state& depth_stencil_state,
                                       const OutVertex&           v0,
                                       const OutVertex&           v1,
                                       const OutVertex&           v2)
    {
      using traits = out_vertex_traits<OutVertex>;

      // Compute triangle equations
      const triangle_equations<OutVertex> eqn{v0, v1, v2};

      // Check if triangle is backfacing
      if (eqn.area2 <= 0.0F)
      {
        return;
      }

      const auto* t = &v0;
      const auto* m = &v1;
      const auto* b = &v2;

      // Sort vertices from top to bottom
      if (t->sv_pos.y > m->sv_pos.y)
      {
        std::swap(t, m);
      }

      if (m->sv_pos.y > b->sv_pos.y)
      {
        std::swap(m, b);
      }

      if (t->sv_pos.y > m->sv_pos.y)
      {
        std::swap(t, m);
      }

      const auto dy = b->sv_pos.y - t->sv_pos.y;
      const auto iy = m->sv_pos.y - t->sv_pos.y;

      if (m->sv_pos.y == t->sv_pos.y)
      {
        const auto* l = m;
        const auto* r = t;

        if (l->sv_pos.x > t->sv_pos.x)
        {
          std::swap(l, r);
        }

        draw_top_flat_triangle(scissor_rect,
                               pixel_shader,
                               color_targets,
                               depth_target,
                               blend_state,
                               depth_stencil_state,
                               eqn,
                               *l,
                               *r,
                               *b);
      }
      else if (m->sv_pos.y == b->sv_pos.y)
      {
        const auto* l = m;
        const auto* r = b;

        if (l->sv_pos.x > r->sv_pos.x)
        {
          std::swap(l, r);
        }

        draw_bottom_flat_triangle(scissor_rect,
                                  pixel_shader,
                                  color_targets,
                                  depth_target,
                                  blend_state,
                                  depth_stencil_state,
                                  eqn,
                                  *t,
                                  *l,
                                  *r);
      }
      else
      {
        const auto mpos = m->sv_pos;
        const auto tpos = t->sv_pos;
        const auto bpos = b->sv_pos;

        OutVertex bary_vertex;

        const vec4 interpolated_pos{tpos.x + ((bpos.x - tpos.x) / dy) * iy,
                                    mpos.y,
                                    tpos.z + ((bpos.z - tpos.z) / dy) * iy,
                                    tpos.w + ((bpos.w - tpos.w) / dy) * iy};

        // Manual barycentric interpolation for the position, because not all components are interpolated.
        interpolated_pos.assign_to(bary_vertex.sv_pos);

        // Barycentric interpolation for vertex attributes
        {
          auto* dst_ptr = traits::attrib_ptr(bary_vertex);
          auto* lhs_ptr = traits::attrib_ptr(*t);
          auto* rhs_ptr = traits::attrib_ptr(*b);

          for (size_t i = 0; i < traits::num_attribs; ++i, ++dst_ptr, ++lhs_ptr, ++rhs_ptr)
          {
            const auto l = *lhs_ptr;
            const auto r = *rhs_ptr;
            *dst_ptr     = l + ((r - l) / dy) * iy;
          }
        }

        const auto* l = m;
        const auto* r = &bary_vertex;

        if (l->sv_pos.x > r->sv_pos.x)
        {
          std::swap(l, r);
        }

        draw_bottom_flat_triangle(scissor_rect,
                                  pixel_shader,
                                  color_targets,
                                  depth_target,
                                  blend_state,
                                  depth_stencil_state,
                                  eqn,
                                  *t,
                                  *l,
                                  *r);

        draw_top_flat_triangle(scissor_rect,
                               pixel_shader,
                               color_targets,
                               depth_target,
                               blend_state,
                               depth_stencil_state,
                               eqn,
                               *l,
                               *r,
                               *b);
      }
    }

    template <out_vertex OutVertex,
              typename ColorImages,
              depth_format                                DepthFormat,
              pixel_shader<OutVertex, ColorImages::count> PixelShader>
    static void draw_triangle(const rect&                scissor_rect,
                              const PixelShader&         pixel_shader,
                              ColorImages&               color_targets,
                              image2d<DepthFormat>&      depth_target,
                              const blend_state&         blend_state,
                              const depth_stencil_state& depth_stencil_state,
                              const OutVertex&           v0,
                              const OutVertex&           v1,
                              const OutVertex&           v2)
    {
      draw_triangle_spanwise(scissor_rect,
                             pixel_shader,
                             color_targets,
                             depth_target,
                             blend_state,
                             depth_stencil_state,
                             v0,
                             v1,
                             v2);
    }

    template <out_vertex OutVertex,
              index_type IndexType,
              typename ColorImages,
              depth_format                                DepthFormat,
              pixel_shader<OutVertex, ColorImages::count> PixelShader>
    static void draw_triangle_list(const rect&                   scissor_rect,
                                   const PixelShader&            pixel_shader,
                                   ColorImages&                  color_targets,
                                   image2d<DepthFormat>&         depth_target,
                                   const blend_state&            blend_state,
                                   const depth_stencil_state&    depth_stencil_state,
                                   const std::vector<OutVertex>& vertices,
                                   const std::vector<IndexType>& indices)
    {
      for (size_t i = 0; (i + 3) <= indices.size(); i += 3)
      {
        if (indices[i] == no_index<IndexType>)
        {
          continue;
        }

        const auto& v0 = vertices[indices[i]];
        const auto& v1 = vertices[indices[i + 1]];
        const auto& v2 = vertices[indices[i + 2]];

        draw_triangle(scissor_rect,
                      pixel_shader,
                      color_targets,
                      depth_target,
                      blend_state,
                      depth_stencil_state,
                      v0,
                      v1,
                      v2);
      }
    }

    template <out_vertex OutVertex,
              index_type IndexType,
              size_t     VertexCacheSize,
              typename ColorImages,
              depth_format                                DepthFormat,
              pixel_shader<OutVertex, ColorImages::count> PixelShader>
    static void draw_primitives(const rect&                                          scissor_rect,
                                const PixelShader&                                   pixel_shader,
                                ColorImages&                                         color_targets,
                                image2d<DepthFormat>&                                depth_target,
                                const blend_state&                                   blend_state,
                                const depth_stencil_state&                           depth_stencil_state,
                                raster_cache<OutVertex, IndexType, VertexCacheSize>& cache,
                                primitive_topology                                   topology)
    {
      switch (topology)
      {
        case primitive_topology::point:
          draw_point_list(scissor_rect,
                          pixel_shader,
                          color_targets,
                          depth_target,
                          blend_state,
                          depth_stencil_state,
                          cache.vertices_out,
                          cache.indices_out);
          break;
        case primitive_topology::line:
          draw_line_list(scissor_rect,
                         pixel_shader,
                         color_targets,
                         depth_target,
                         blend_state,
                         depth_stencil_state,
                         cache.vertices_out,
                         cache.indices_out);
          break;
        case primitive_topology::triangle_list:
          draw_triangle_list(scissor_rect,
                             pixel_shader,
                             color_targets,
                             depth_target,
                             blend_state,
                             depth_stencil_state,
                             cache.vertices_out,
                             cache.indices_out);
          break;
      }
    }

    template <out_vertex OutVertex, index_type IndexType, size_t VertexCacheSize>
    static void cull_triangles(const rasterizer_state&                              rasterizer_state,
                               raster_cache<OutVertex, IndexType, VertexCacheSize>& cache)
    {
      const auto& vertices = cache.vertices_out;
      auto&       indices  = cache.indices_out;

      for (size_t i = 0; i + 3 <= indices.size(); i += 3)
      {
        if (indices[i] == no_index<IndexType>)
        {
          continue;
        }

        const auto& v0 = vertices[indices[i]];
        const auto& v1 = vertices[indices[i + 1]];
        const auto& v2 = vertices[indices[i + 2]];

        const auto v0x = v0.sv_pos.x;
        const auto v0y = v0.sv_pos.y;

        const auto v1x = v1.sv_pos.x;
        const auto v1y = v1.sv_pos.y;

        const auto v2x = v2.sv_pos.x;
        const auto v2y = v2.sv_pos.y;

        const auto facing = (v0x - v1x) * (v2y - v1y) - (v2x - v1x) * (v0y - v1y);

        if (facing < 0.0F)
        {
          if (rasterizer_state.cull_mode == cull_mode::cull_clockwise)
          {
            indices[i]     = no_index<IndexType>;
            indices[i + 1] = no_index<IndexType>;
            indices[i + 2] = no_index<IndexType>;
          }
        }
        else
        {
          if (rasterizer_state.cull_mode == cull_mode::cull_counter_clockwise)
          {
            indices[i]     = no_index<IndexType>;
            indices[i + 1] = no_index<IndexType>;
            indices[i + 2] = no_index<IndexType>;
          }
          else
          {
            std::swap(indices[i], indices[i + 2]);
          }
        }
      }
    }

    template <out_vertex OutVertex,
              index_type IndexType,
              size_t     VertexCacheSize,
              typename ColorImages,
              depth_format                                DepthFormat,
              pixel_shader<OutVertex, ColorImages::count> PixelShader>
    static void process_primitives(const viewport&                                      viewport,
                                   const rect&                                          scissor_rect,
                                   const rasterizer_state&                              rasterizer_state,
                                   const PixelShader&                                   pixel_shader,
                                   ColorImages&                                         color_targets,
                                   image2d<DepthFormat>&                                depth_target,
                                   const blend_state&                                   blend_state,
                                   const depth_stencil_state&                           depth_stencil_state,
                                   raster_cache<OutVertex, IndexType, VertexCacheSize>& cache,
                                   primitive_topology                                   topology)
    {
      clip_primitives(cache, topology);
      transform_vertices(viewport, cache);

      if (topology == primitive_topology::triangle_list)
      {
        cull_triangles(rasterizer_state, cache);
      }

      draw_primitives(scissor_rect,
                      pixel_shader,
                      color_targets,
                      depth_target,
                      blend_state,
                      depth_stencil_state,
                      cache,
                      topology);
    }
  } // namespace details

  template <in_vertex  InVertex,
            out_vertex OutVertex,
            index_type IndexType,
            typename ColorImages,
            depth_format                                DepthFormat,
            vertex_shader<InVertex, OutVertex>          VertexShader,
            pixel_shader<OutVertex, ColorImages::count> PixelShader,
            size_t                                      VertexCacheSize>
  void draw_indexed(ColorImages                                          color_targets,
                    image2d<DepthFormat>&                                depth_target,
                    const VertexShader&                                  vertex_shader,
                    const PixelShader&                                   pixel_shader,
                    const viewport&                                      viewport,
                    const rect&                                          scissor_rect,
                    const blend_state&                                   blend_state,
                    const depth_stencil_state&                           depth_stencil_state,
                    const rasterizer_state&                              rasterizer_state,
                    primitive_topology                                   topology,
                    raster_cache<OutVertex, IndexType, VertexCacheSize>& cache,
                    vertex_view<InVertex>                                vertices,
                    index_view<IndexType>                                indices,
                    size_t                                               start_index,
                    size_t                                               index_count,
                    int                                                  base_vertex_index)
  {
    static_assert(sizeof(InVertex) >= sizeof(float), "input vertex must have at least one float");
    static_assert(sizeof(OutVertex) >= sizeof(rgba32f), "output vertex must have at least one vec4");

    assert(indices.size() % 3 == 0);
    assert(*std::max_element(indices.begin(), indices.end()) < vertices.size());

    cache.clear();

    const size_t end_index = start_index + index_count;

    for (size_t i = start_index; i < end_index; ++i)
    {
      const auto index        = static_cast<IndexType>(static_cast<int>(indices[i]) + base_vertex_index);
      auto       output_index = cache.vcache.lookup(index);

      if (output_index != no_index<IndexType>)
      {
        cache.indices_out.push_back(output_index);
      }
      else
      {
        output_index = static_cast<IndexType>(cache.vertices_out.size());
        cache.indices_out.push_back(output_index);

        const auto vOut = vertex_shader.execute(vertices[index]);
        cache.vertices_out.push_back(vOut);

        cache.vcache.set(index, output_index);
      }
    }

    details::process_primitives(viewport,
                                scissor_rect,
                                rasterizer_state,
                                pixel_shader,
                                color_targets,
                                depth_target,
                                blend_state,
                                depth_stencil_state,
                                cache,
                                topology);
  }

  template <typename ColorImages, single_pixel_pixel_shader<ColorImages::count> PixelShader>
  inline void for_each_pixel(ColorImages        color_targets,
                             const PixelShader& pixel_shader,
                             const blend_state& blend_state,
                             uv_origin          uv_origin)
  {
    const auto width   = static_cast<int>(color_targets.width());
    const auto height  = static_cast<int>(color_targets.height());
    const auto widthf  = color_targets.widthf();
    const auto heightf = color_targets.heightf();

    const auto pixel_count = width * height;

#if defined(NDEBUG)
#pragma omp parallel for
#endif
    for (int i = 0; i < pixel_count; ++i)
    {
      const auto x = i % width;
      const auto y = i / width;

      const auto src_colors = pixel_shader.execute(single_pixel_ps_input{
          .x = static_cast<unsigned>(x),
          .y = static_cast<unsigned>(y),
          .u = x / widthf,
          .v = uv_origin == uv_origin::top_left ? y / heightf : 1.0F - (y / heightf),
      });

      details::write_pixel_tuple(color_targets.images(), src_colors, blend_state, x, y);
    }
  }
} // namespace ras
