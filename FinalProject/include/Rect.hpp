#ifndef RECT_HPP
#define RECT_HPP

#include <algorithm>
#include <array>

namespace prj
{
   // Rectangle.
   template<typename T>
   struct Rect
   {
      // Sides of the rectangle.
      enum class Side
      {
         right,
         top,
         left,
         bottom,
         tot
      };

      /********** CONSTRUCTORS **********/
      static constexpr int fields{ 4 }; // Number of fields.
      constexpr Rect() = default;
      constexpr Rect(const T px, const T py, const T pw, const T ph) :
         x{ px }, y{ py }, w{ pw }, h{ ph } {}
      constexpr explicit Rect(const std::array<T, fields>& data) :
         x{ data[0] }, y{ data[1] }, w{ data[2] }, h{ data[3] } {}

      /********** OPERATORS **********/
      bool operator!=(const Rect<T>& other)
      {
         return x != other.x || y != other.y || w != other.w || h != other.h;
      }

      /********** METHODS **********/
      // Check whether a point is inside the rectangle or not.
      [[nodiscard]] constexpr bool contains(const T& ptX, const T& ptY) const
      {
         return (ptX >= x && ptX <= x + w && ptY >= y && ptY <= y + h);
      }
      // Check whether another rectangle is inside this one or not.
      [[nodiscard]] constexpr bool contains(const Rect<T>& rect) const
      {
         return rect.x >= x && rect.x + rect.w <= x + w && rect.y >= y && rect.y + rect.h <= y + h;
      }

      // Extend the rectangle on one side up to a limit.
      constexpr void extend(Side side, const T& amount, const T& limit)
      {
         switch (side)
         {
         case Side::right:
         {
            w += amount;
            T diff{ x + w - limit };
            if (diff > 0) w -= diff;
            break;
         }
         case Side::top:
         {
            T newY{ y - amount };
            if (newY < limit) newY = limit;
            h += (y - newY);
            y = newY;
            break;
         }
         case Side::left:
         {
            T newX{ x - amount };
            if (newX < limit) newX = limit;
            w += (x - newX);
            x = newX;
            break;
         }
         case Side::bottom:
         {
            h += amount;
            T diff{ y + h - limit };
            if (diff > 0) h -= diff;
            break;
         }
         default:
            break;
         }
      }

      // Compute the extension rectangle on the specified side.
      [[nodiscard]] constexpr Rect<T> getExtension(Side side, const T& amount, const T& limit) const
      {
         Rect<T> result;
         switch (side)
         {
         case Side::right:
         {
            result.x = x + w;
            result.y = y;
            result.h = h;
            result.w = amount;
            T diff{ result.x + result.w - limit };
            if (diff > 0) result.w -= diff;
         }
         break;
         case Side::top:
         {
            result.x = x;
            result.w = w;
            result.y = y - amount;
            if (result.y < limit) result.y = limit;
            result.h = y - result.y;
         }
         break;
         case Side::left:
         {
            result.y = y;
            result.h = h;
            result.x = x - amount;
            if (result.x < limit) result.x = limit;
            result.w = x - result.x;
         }
         break;
         case Side::bottom:
         {
            result.x = x;
            result.y = y + h;
            result.w = w;
            result.h = result.y + amount;
            T diff{ result.y + result.h - limit };
            if (diff > 0) result.h -= diff;
         }
         break;
         default:
            break;
         }
         return result;
      }

      // Checks whether the current rectangle overlaps a target one by at least ratio * smallest area.
      [[nodiscard]] constexpr Rect<T> overlaps(const Rect<T>& rect, float ratio)
      {
         T thisMaxX{ x + w };
         T thisMaxY{ y + h };
         T otherMaxX{ rect.x + rect.w };
         T otherMaxY{ rect.y + rect.h };

         if (thisMaxX <= rect.x || otherMaxX <= x || thisMaxY <= rect.y || otherMaxY <= y)
            return Rect<int>(0, 0, 0, 0);

         T thisArea{ w * h };
         T otherArea{ rect.w * rect.h };

         std::array orderedX{
            x, thisMaxX, rect.x, otherMaxX
         };
         std::sort(orderedX.begin(), orderedX.end());
         std::array orderedY{
            y, thisMaxY, rect.y, otherMaxY
         };
         std::sort(orderedY.begin(), orderedY.end());

         T overlapW{ orderedX[2] - orderedX[1] };
         T overlapH{ orderedY[2] - orderedY[1] };
         T overlapArea{ overlapW * overlapH };

         T minArea{ std::min(thisArea, otherArea) };
         if (overlapArea >= ratio * minArea)
            return Rect<T>(orderedX[0], orderedY[0], orderedX[3] - orderedX[0], orderedY[3] - orderedY[0]);

         return Rect<T>(0, 0, 0, 0);
      }

      // Position of the top-left vertex.
      T x{};
      T y{};
      // Size of the rectangle.
      T w{};
      T h{};
   };
} // namespace prj

#endif