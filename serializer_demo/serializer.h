#ifndef SERIALIZER_H_
#define SERIALIZER_H_

#include <stdint.h>
#include <string>
#include <cassert>
#include <algorithm>

#include <vector>
#include <deque>
#include <list>
#include <set>
#include <map>
#include <iostream>

#include <exception>
#include <stdexcept>

namespace EndianSwapper
{
class SwapByteBase
{
public:
    static bool ShouldSwap()
    {
        static const uint16_t swapTest = 1;
        return (*((char *) &swapTest) == 1);
    }

    static void SwapBytes(uint8_t &v1, uint8_t &v2)
    {
        uint8_t tmp = v1;
        v1 = v2;
        v2 = tmp;
    }
};

template<class T, int S>
class SwapByte : public SwapByteBase
{
public:
    static T Swap(T v)
    {
        assert(false); // Shoud not be here...
        return v;
    }
};

template<class T>
class SwapByte<T, 1> : public SwapByteBase
{
public:
    static T Swap(T v)
    {
        return v;
    }
};

template<class T>
class SwapByte<T, 2> : public SwapByteBase
{
public:
    static T Swap(T v)
    {
        if (ShouldSwap())
        {
            return (v >> 8) | (v << 8);
        }
        return v;
    }
};

template<class T>
class SwapByte<T, 4> : public SwapByteBase
{
public:
    static T Swap(T v)
    {
        if (ShouldSwap())
        {
            return (SwapByte<uint16_t, 2>::Swap((uint32_t) v & 0xffff) << 16) |
                   (SwapByte<uint16_t, 2>::Swap(((uint32_t) v & 0xffff0000) >> 16));
        }
        return v;
    }
};

template<class T>
class SwapByte<T, 8> : public SwapByteBase
{
public:
    static T Swap(T v)
    {
        if (ShouldSwap())
        {
            return (((uint64_t) SwapByte<uint32_t, 4>::Swap((uint32_t) (v & 0xffffffffull))) << 32) |
                   (SwapByte<uint32_t, 4>::Swap((uint32_t) (v >> 32)));
        }
        return v;
    }
};

template<>
class SwapByte<float, 4> : public SwapByteBase
{
public:
    static float Swap(float v)
    {
        union
        {
            float f;
            uint8_t c[4];
        };
        f = v;
        if (ShouldSwap())
        {
            SwapBytes(c[0], c[3]);
            SwapBytes(c[1], c[2]);
        }
        return f;
    }
};

template<>
class SwapByte<double, 8> : public SwapByteBase
{
public:
    static double Swap(double v)
    {
        union
        {
            double f;
            uint8_t c[8];
        };
        f = v;
        if (ShouldSwap())
        {
            SwapBytes(c[0], c[7]);
            SwapBytes(c[1], c[6]);
            SwapBytes(c[2], c[5]);
            SwapBytes(c[3], c[4]);
        }
        return f;
    }
};
};


template <class T>
struct valueTraits {
    typedef T value_type;
};


template<class STREAM_TYPE>
class Archive
{
public:
    Archive(STREAM_TYPE &stream) : m_stream(stream)
    {
    }

public:
    template<class T>
    const Archive &operator<<(const T &v) const
    {
        *this & v;
        return *this;
    }

    template<class T>
    Archive &operator>>(T &v)
    {
        *this & v;
        return *this;
    }

    template<class T>
    int deserializeTo(T &v)
    {
        //try
        {
            *this >> v;
            return 0;
        } /*catch (std::bad_alloc &)
        {
            ERROR("caught bad alloc exception!");
            LOGE("bvt", "caught bad alloc exception!");
            return 1;
        } catch (...)
        {
            ERROR("unknown exception!");
            LOGE("bvt", "unknown exception!");
            return 1;
        }*/
    }

public:
    template<class T>
    Archive &operator&(T &v)
    {
        v.Serialize(*this);
        return *this;
    }

    template<class T>
    const Archive &operator&(const T &v) const
    {
        ((T &) v).Serialize(*this);
        return *this;
    }

    template<class T, size_t N>
    Archive &operator&(T (&v)[N])
    {
        uint32_t len;
        *this & len;
        for (size_t i = 0; i < N; ++i)
        {
            *this & v[i];
        }
        return *this;
    }

    template<class T, size_t N>
    const Archive &operator&(const T (&v)[N]) const
    {
        uint32_t len = N;
        *this & len;
        for (size_t i = 0; i < N; ++i)
        {
            *this & v[i];
        }
        return *this;
    }

#define SERIALIZER_FOR_POD(type) \
        Archive& operator&(type& v) \
        { \
            m_stream.read((char*)&v, sizeof(type)); \
            if (!m_stream) { \
                v = 0; \
                return *this; \
            } \
            v = Swap(v); \
            return *this; \
        } \
        const Archive& operator&(type v) const \
        { \
            v = Swap(v); \
            m_stream.write((const char*)&v, sizeof(type)); \
            return *this; \
        }

    SERIALIZER_FOR_POD(bool);

    SERIALIZER_FOR_POD(char);

    SERIALIZER_FOR_POD(unsigned char);

    SERIALIZER_FOR_POD(short);

    SERIALIZER_FOR_POD(unsigned short);

    SERIALIZER_FOR_POD(int);

    SERIALIZER_FOR_POD(unsigned int);

    SERIALIZER_FOR_POD(long);

    SERIALIZER_FOR_POD(unsigned long);

    SERIALIZER_FOR_POD(long long);

    SERIALIZER_FOR_POD(unsigned long long);

    SERIALIZER_FOR_POD(float);

    SERIALIZER_FOR_POD(double);

#define SERIALIZER_FOR_VECTOR_VALUE_ONE_BYTE(type) \
    template<class T> \
    void processVector(std::vector<T> &v, type) \
    { \
        uint32_t len; \
        *this & len; \
        if (len <= 0) return; \
        if (len >= 1e8) return; \
        v.resize(len); \
        m_stream.read((char *) &(v[0]), sizeof(T) * len); \
    }

#define SERIALIZER_FOR_VECTOR_VALUE_MULTI_BYTE(type) \
    template<class T> \
    void processVector(std::vector<T> &v, type) \
    { \
        uint32_t len; \
        *this & len; \
        if (len <= 0) return; \
        if (len >= 1e8) return; \
        v.resize(len); \
        for (uint32_t i = 0; i < len; ++i) \
        { \
            T value; \
            *this & value; \
            v[i] = value; \
        } \
    }

    SERIALIZER_FOR_VECTOR_VALUE_ONE_BYTE(bool);

    SERIALIZER_FOR_VECTOR_VALUE_ONE_BYTE(char);

    SERIALIZER_FOR_VECTOR_VALUE_ONE_BYTE(unsigned char);

    SERIALIZER_FOR_VECTOR_VALUE_MULTI_BYTE(short);

    SERIALIZER_FOR_VECTOR_VALUE_MULTI_BYTE(unsigned short);

    SERIALIZER_FOR_VECTOR_VALUE_MULTI_BYTE(int);

    SERIALIZER_FOR_VECTOR_VALUE_MULTI_BYTE(unsigned int);

    SERIALIZER_FOR_VECTOR_VALUE_MULTI_BYTE(long);

    SERIALIZER_FOR_VECTOR_VALUE_MULTI_BYTE(unsigned long);

    SERIALIZER_FOR_VECTOR_VALUE_MULTI_BYTE(long long);

    SERIALIZER_FOR_VECTOR_VALUE_MULTI_BYTE(unsigned long long);

    SERIALIZER_FOR_VECTOR_VALUE_MULTI_BYTE(float);

    SERIALIZER_FOR_VECTOR_VALUE_MULTI_BYTE(double);


    template<class T>
    Archive &operator&(std::vector<T> &v)
    {
        typedef typename valueTraits<T>::value_type value_type;
        processVector(v, value_type());
        return *this;
    }


    template<class T>
    const Archive &operator&(const std::vector<T> &v) const
    {
        uint32_t len = v.size();
        *this & len;
        for (typename std::vector<T>::const_iterator it = v.begin(); it != v.end(); ++it)
        {
            *this & *it;
        }
        return *this;
    }


#define SERIALIZER_FOR_STL(type) \
        template <class T> \
        Archive& operator&(type<T>& v) \
        { \
            uint32_t len; \
            *this & len; \
            for(uint32_t i = 0; i < len; ++i) \
            { \
                T value; \
                *this & value; \
                v.insert(v.end(), value); \
            } \
            return *this; \
        } \
        template <class T> \
        const Archive& operator&(const type<T>& v) const \
        { \
            uint32_t len = v.size(); \
            *this & len; \
            for(typename type<T>::const_iterator it = v.begin(); it != v.end(); ++it) \
            *this & *it; \
            return *this; \
        }

#define SERIALIZER_FOR_STL2(type) \
        template <class T1, class T2> \
        Archive& operator&(type<T1, T2>& v) \
        { \
            uint32_t len; \
            *this & len; \
            for(uint32_t i = 0; i < len; ++i) \
            { \
                std::pair<T1, T2> value; \
                *this & value; \
                v.insert(v.end(), value); \
            } \
            return *this; \
        } \
        template <class T1, class T2> \
        const Archive& operator&(const type<T1, T2>& v) const \
        { \
            uint32_t len = v.size(); \
            *this & len; \
            for(typename type<T1, T2>::const_iterator it = v.begin(); it != v.end(); ++it) \
            *this & *it; \
            return *this; \
        }

    SERIALIZER_FOR_STL(std::deque);

    SERIALIZER_FOR_STL(std::list);

    SERIALIZER_FOR_STL(std::set);

    SERIALIZER_FOR_STL(std::multiset);

    SERIALIZER_FOR_STL2(std::map);

    SERIALIZER_FOR_STL2(std::multimap);

    template<class T1, class T2>
    Archive &operator&(std::pair<T1, T2> &v)
    {
        *this & v.first & v.second;
        return *this;
    }

    template<class T1, class T2>
    const Archive &operator&(const std::pair<T1, T2> &v) const
    {
        *this & v.first & v.second;
        return *this;
    }

    Archive &operator&(std::string &v)
    {
        uint32_t len;
        *this & len;
        v.clear();
        char buffer[4096];
        uint32_t toRead = len;
        while (toRead != 0)
        {
            uint32_t l = std::min(toRead, (uint32_t) sizeof(buffer));
            m_stream.read(buffer, l);
            if (!m_stream)
            {
                return *this;
            }
            v += std::string(buffer, l);
            toRead -= l;
        }
        return *this;
    }

    const Archive &operator&(const std::string &v) const
    {
        uint32_t len = v.length();
        *this & len;
        m_stream.write(v.c_str(), len);
        return *this;
    }

private:
    template<class T>
    T Swap(const T &v) const
    {
        return EndianSwapper::SwapByte<T, sizeof(T)>::Swap(v);
    }

private:
    STREAM_TYPE &m_stream;
};

#endif // SERIALIZER_H_
