/* $Header: /source/ArgoSoft5/ArgoCore/Stopwatch.cpp,v 1.1 2007/03/28 06:07:08 ads Exp $ */
// Stopwatch.cpp

//#include "common.h"

#include "Stopwatch.h"

#include <math.h>   // for floor()
#include <stdio.h>  // for NULL


// Begin machine-dependent section...

#ifdef PLATFORM_GCC

inline static void normalize_time(struct timeval *t)
{
    long ds;
    /* ensure that tv_usec is between 0 and 1000000 */
    if (t->tv_usec > 1000000) {
        ds = t->tv_usec / 1000000;
        t->tv_sec += ds;
        t->tv_usec -= ds*1000000;
    } else if (t->tv_usec < 0) {
        ds = (-t->tv_usec) / 1000000;
        t->tv_sec -= ds;
        t->tv_usec += ds*1000000;
        if (t->tv_usec < 0) {
            t->tv_sec -= 1;
            t->tv_usec += 1000000;
        }
    }
}
inline static struct timeval GET_TIME(bool useWorldTime)
{
    if (useWorldTime) {
        struct timeval t;
        gettimeofday(&t, NULL);
        return t;
    } else {
        struct timeval t;
        struct rusage ru;
        getrusage(RUSAGE_SELF, &ru);
        t.tv_sec = ru.ru_utime.tv_sec + ru.ru_stime.tv_sec;
        t.tv_usec = ru.ru_utime.tv_usec + ru.ru_stime.tv_usec;
        normalize_time(&t);
        return t;
    }
}
inline static struct timeval ADD_TIME(struct timeval t2, struct timeval t1)
{
    struct timeval t;
    t.tv_sec = t2.tv_sec + t1.tv_sec;
    t.tv_usec = t2.tv_usec + t1.tv_usec;
    normalize_time(&t);
    return t;
}
inline static struct timeval SUBTRACT_TIME(struct timeval t2, struct timeval t1)
{
    struct timeval t;
    t.tv_sec = t2.tv_sec - t1.tv_sec;
    t.tv_usec = t2.tv_usec - t1.tv_usec;
    normalize_time(&t);
    return t;
}
inline static struct timeval ZERO_TIME(void)
{
    struct timeval t;
    t.tv_sec = 0;
    t.tv_usec = 0;
    return t;
}
inline static void EXTRACT_TIME_PIECES(struct timeval t, long int& sec, long int& usec)
{
    normalize_time(&t);
    sec = t.tv_sec;
    usec = t.tv_usec;
}
inline static double TIME_TO_SEC(struct timeval t)
{
    return (double)t.tv_sec + ( (double)t.tv_usec / 1.0e6 );
}
#define USE_WORLD_TIME(b) (b)


#elif defined(PLATFORM_WIN)


inline static ULARGE_INTEGER GET_TIME(bool useWorldTime)
{
    if (useWorldTime) {
        SYSTEMTIME s;
        FILETIME f;
        GetSystemTime(&s);
        SystemTimeToFileTime(&s, &f);
        ULARGE_INTEGER ftime;
        ftime.HighPart = f.dwHighDateTime;
        ftime.LowPart = f.dwLowDateTime;
        return ftime;
    } else {
        FILETIME CreationTime, ExitTime, KernelTime, UserTime;
        ULARGE_INTEGER utime, ktime, ut;
        GetProcessTimes(GetCurrentProcess(), &CreationTime, &ExitTime, &KernelTime, &UserTime);
        utime.HighPart = UserTime.dwHighDateTime;
        utime.LowPart = UserTime.dwLowDateTime;
        ktime.HighPart = KernelTime.dwHighDateTime;
        ktime.LowPart = KernelTime.dwLowDateTime;
        ut.QuadPart = utime.QuadPart + ktime.QuadPart;
        return ut;
    }
}
inline static ULARGE_INTEGER ADD_TIME(ULARGE_INTEGER t2, ULARGE_INTEGER t1)
{
    ULARGE_INTEGER ut;
    ut.QuadPart = t2.QuadPart + t1.QuadPart;
    return ut;
}
inline static ULARGE_INTEGER SUBTRACT_TIME(ULARGE_INTEGER t2, ULARGE_INTEGER t1)
{
    ULARGE_INTEGER ut;
    ut.QuadPart = t2.QuadPart - t1.QuadPart;
    return ut;
}
inline static ULARGE_INTEGER ZERO_TIME(void)
{
    ULARGE_INTEGER ut;
    ut.QuadPart = (ULONGLONG)0;
    return ut;
}
inline static void EXTRACT_TIME_PIECES(ULARGE_INTEGER t, long int& sec, long int& usec)
{
    ULONGLONG llSec = (t.QuadPart / (ULONGLONG)(10000000));
    ULONGLONG llUsec = (t.QuadPart / (ULONGLONG)10) % (ULONGLONG)1000000;
    sec = (long int)llSec;
    usec = (long int)llUsec;
}
inline static bool USE_WORLD_TIME(bool useWorldTime)
{
    DWORD dwVersion = GetVersion();
    if (dwVersion < 0x80000000) {
        // Windows NT, 2000, or better.
        return useWorldTime;
    } else {
        // Win32s or Windows 95/98/ME.
        return true;
    }
}
// The FILETIME data structure stores time 100-nanosecond intervals, in a 64-bit number
// (split into two 32-bit words).
const double lowTimeAdj = 1.0e-7;           // 100ns
const double highTimeAdj = 4294967296.0e-7; // (2^32) * 100ns
inline static double TIME_TO_SEC(ULARGE_INTEGER ut)
{
    return (double(ut.HighPart) * highTimeAdj) + (double(ut.LowPart) * lowTimeAdj);
}


#else // Only use standard ANSI C functions


/* If you define STOPWATCH_FINE, then the stopwatch functions
 * use clock(), which allows timing of events shorter than
 * one second. However, trying to time events that take several hours
 * may cause integer overflows.
 *
 * If you do not define STOPWATCH_FINE, then the stopwatch functions
 * use time(), which has a resolution of one second.
 *
 * With STOPWATCH_FINE,
 */
#ifdef STOPWATCH_FINE
#define GET_TIME(x) clock()
#define TIME_TO_SEC(t) ( (double)t / (double)CLOCKS_PER_SEC )
#else
#define GET_TIME(x) time(NULL)
#define TIME_TO_SEC(t) ( (double)t )
#endif

#define ADD_TIME(t2, t1) ( (t2) + (t1) )
#define SUBTRACT_TIME(t2, t1) ( (t2) - (t1) )
#define ZERO_TIME() (0)

static inline void EXTRACT_TIME_PIECES(clock_t t, long int& sec, long int& usec)
{
#ifdef STOPWATCH_FINE
    sec = t/CLOCKS_PER_SEC;
    long int usecTicks = (t - CLOCKS_PER_SEC*sec);
    // Note: If CLOCKS_PER_SEC is large (approaching or exceeding 1000000),
    // then this won't work well. The value of usec will be under-estimated or zero.
    // However, performing multiplication and then division will cause integer overflow.
    usec = usecTicks * (1000000/CLOCKS_PER_SEC);
#else
    sec = t;
    usec = 0;
#endif
}

// The standard ANSI C functions do not differentiate between
// process time and world time. According to the man pages,
// time() measures world time, while clock() measures CPU time.
#ifdef STOPWATCH_FINE
#define USE_WORLD_TIME(b) (false)
#else
#define USE_WORLD_TIME(b) (true)
#endif


#endif

// ... end machine-dependent section






/**
 * Convert this from sec, usec representation to a double.
 */
ElapsedTime::operator double() const
{
    return double(tv_sec) + double(tv_usec)*1.0e-6;
}

/**
 * Constructor: Convert from a double representation to a sec, usec representation
 * stored in this. Note: negative times are truncated to 0.
 */
ElapsedTime::ElapsedTime(double t)
{
    if (t <= 0.0) {
        tv_sec = 0;
        tv_usec = 0;
    } else {
        double dSec = floor(t);
        tv_sec = (long int)dSec;
        double dUsec = (t - dSec)*1.0e6;
        tv_usec = (long int)dUsec;
    }
}

/// Normalize the time representation.
/**
 * Adjust the time sto that tv_usec is in [0, 999999].
 */
void ElapsedTime::normalize()
{
    if (isNormalized()) {
        return;
    }
    long int delta_s= tv_sec/1000000;
    tv_sec += delta_s;
    tv_usec -= delta_s*1000000;
    // Note: We multiply rather than using % on tv_usec,
    // because % is not safe with -ve numbers here. Whether % returns a + or -
    // result is implementation-defined, and the "wrong" one will cause an off-by-one
    // error in the number of seconds adjusted for.

    // tv_usec is in [-999999, 999999]. Now make it positive.
    if (tv_usec < 0) {
        tv_sec--;
        tv_usec += 1000000;
    }
}


/// less than operator
bool ElapsedTime::operator<(const ElapsedTime& other) const
{
    ElapsedTime t1(tv_sec, tv_usec);
    ElapsedTime t2(other);
    t1.normalize();
    t2.normalize();
    return ( (t1.tv_sec < t2.tv_sec) || ( (t1.tv_sec == t2.tv_sec) && (t1.tv_usec < t2.tv_usec) ));
}

/// equality operator
bool ElapsedTime::operator==(const ElapsedTime& other) const
{
    ElapsedTime t1(tv_sec, tv_usec);
    ElapsedTime t2(other);
    t1.normalize();
    t2.normalize();
    return ((t1.tv_sec == t2.tv_sec) && (t1.tv_usec == t2.tv_usec));
}

/// addition operator
ElapsedTime ElapsedTime::operator+(const ElapsedTime& other) const
{
    ElapsedTime v1(tv_sec, tv_usec);
    ElapsedTime v2(other);
    v1.normalize();
    v2.normalize();
    v1.tv_sec += v2.tv_sec;
    v1.tv_usec += v2.tv_usec;
    v1.normalize();
    return v1;
}

/// subtraction operator
ElapsedTime ElapsedTime::operator-(const ElapsedTime& other) const
{
    ElapsedTime v1(tv_sec, tv_usec);
    ElapsedTime v2(other);
    v1.normalize();
    v2.normalize();
    v1.tv_sec -= v2.tv_sec;
    v1.tv_usec -= v2.tv_usec;
    v1.normalize();
    return v1;
}






/**
 * Initializes the stopwatch. If the current platform supports measuring
 * non-world time (ie. user plus kernel time), then the 'worldTime'
 * variable determines whether or not that mode will be used. If the
 * platform does not support this, then 'worldTime' is ignored, and
 * is treated as if it were true.
 *
 * @note You can call usingWorldTime() to determine whether a Stopwatch
 * is measuring world or processor time.
 */
Stopwatch::Stopwatch(bool worldTime)
{
    m_useWorldTime = USE_WORLD_TIME(worldTime);
    reset();
}


/// Start the stopwatch.
/**
 * Start the stopwatch. Time will continue to accumulate from the current time on the stopwatch.
 */
void Stopwatch::start()
{
    if (m_running)
        return;
    // else...

    m_start = GET_TIME(m_useWorldTime);
    m_running = true;
}


/// Stop the stopwatch.
void Stopwatch::stop()
{
    if (!m_running)
        return;
    // else...

    m_elapsed = ADD_TIME(m_elapsed, SUBTRACT_TIME(GET_TIME(m_useWorldTime), m_start));
    m_running = false;
}


/// Reset to zero and stop.
/**
 * Reset the stopwatch to zero time, and stop it from running.
 */
void Stopwatch::reset()
{
    m_elapsed = ZERO_TIME();
    m_start = ZERO_TIME();
    m_running = false;
}


// Reset to zero and start.
/*
 * Reset the stopwatch to zero time, and start it running.
 */
void Stopwatch::restart()
{
    m_elapsed = ZERO_TIME();
    m_start = GET_TIME(m_useWorldTime);
    m_running = true;
}


/// Get accumulated time.
/**
 * Get the current time on the stopwatch in seconds. If the stopwatch
 * is running, then the elapsed time so far is returned, and it continues
 * running. If it is stopped, then the value at which it stopped is returned.
 * (The resolution of the stopwatch is machine-dependent.)
 *
 * Using the elapsed time as a double may incur rounding errors. To get a
 * more accurate representation of the elapsed time, use the overloaded
 * versions of this function.
 */
double Stopwatch::getElapsed() const
{
    if (m_running) {
        return TIME_TO_SEC( ADD_TIME(m_elapsed,
                            SUBTRACT_TIME(GET_TIME(m_useWorldTime), m_start)));
    } else {
        return TIME_TO_SEC(m_elapsed);
    }
}

/// Get accumulated time in seconds and microseconds.
/**
 * Get the current time on the stopwatch in seconds and microseconds. The number
 * of microseconds will always be less than 1000000. If the stopwatch
 * is running, then the elapsed time so far is returned, and it continues
 * running. If it is stopped, then the value at which it stopped is returned.
 * (The resolution of the stopwatch is machine-dependent, and may be more or
 * less than one microsecond.)
 */
void Stopwatch::getElapsed(long int& sec, long int& usec) const
{
    if (m_running) {
        EXTRACT_TIME_PIECES( ADD_TIME(m_elapsed,
                             SUBTRACT_TIME(GET_TIME(m_useWorldTime), m_start)),
                             sec, usec);
    } else {
        EXTRACT_TIME_PIECES(m_elapsed, sec, usec);
    }
}
