/* $Header: /source/ArgoSoft5/ArgoCore/Stopwatch.h,v 1.1 2007/03/28 06:07:09 ads Exp $ */
/** @file

  Stopwatch.h
  (Stopwatch class)

  This class provides a stopwatch which can be used to time events.

*/


//#include "common.h"
// Automatically detect the PLATFORM_* type (unless requested not to).
#ifndef NO_AUTO_PLATFORM
#if defined(_MSC_VER)
    #define PLATFORM_WIN
#elif defined(__GNUC__)
    #define PLATFORM_GCC
#else
    #define PLATFORM_UNKNOWN
#endif
#endif


#ifndef _STOPWATCH_H_
#define _STOPWATCH_H_

#ifdef PLATFORM_GCC
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <unistd.h>
#elif defined(PLATFORM_WIN)
#include <windows.h>
#include <winsock2.h>
#else
#include <time.h>
#endif


/// time differences
/**
 * This class is used to represent elapsed time in seconds and microseconds.
 */
class ElapsedTime : public timeval {
public:
    ElapsedTime() { tv_sec = 0; tv_usec = 0; }
    ElapsedTime(long int sec, long int usec = 0) { tv_sec = sec; tv_usec = usec; }
    ElapsedTime(double t);
    void normalize();
    inline bool isNormalized() const { return (tv_usec >= 0) && (tv_usec < 1000000); }

    operator double () const;
    bool operator<(const ElapsedTime& other) const;
    bool operator==(const ElapsedTime& other) const;
    ElapsedTime operator+(const ElapsedTime& other) const;
    ElapsedTime operator-(const ElapsedTime& other) const;
};


/// timing functions for real-world and CPU time
/**
 * This class provides a stopwatch which can be used to time events.
 * A Stopwach can measure either real-world or CPU time.
 */
class Stopwatch {
public:
    Stopwatch(bool worldTime);
    ~Stopwatch() { }

    void start();
    void stop();
    void reset();
    void restart();
    double getElapsed() const;
    void getElapsed(long int& sec, long int& usec) const;
    /// Get the current time on the stopwatch in seconds and microseconds.
    void getElapsed(ElapsedTime& elapsed) const {
        getElapsed(elapsed.tv_sec, elapsed.tv_usec);
    }

    /// returns true if using real-world time
    bool usingWorldTime() const { return m_useWorldTime; }

protected:
#ifdef PLATFORM_GCC
    struct timeval m_elapsed;
    struct timeval m_start;

#elif defined(PLATFORM_WIN)
    // Note: There appears to be a bug in MS Visual C++ v6.0.
    // It does not align ULARGE_INTEGERs correctly.
    // The symptoms are that when accessing m_elapsed and/or m_start, it may
    // write values starting at the next 32(?)-byte boundary, thereby clobbering
    // the data in memory after it.
    // To work around this, it is necessary to ensure that these data members
    // are aligned properly. The easiest way to do that is to make sure they are
    // the first variables defined in the class. The other members below
    // (bool m_running and bool m_useWorldTime) are smaller, so they must appear
    // after these variables (m_elapsed and m_start) to avoid problems.
    //
    // This problem probably happens because of the following: According to MSDN
    // documentation, Visual C++ 6 aligns unions according to the requirements of
    // the *first* member. A ULARGE_INTEGER is defined as a union of a struct of
    // two DWORDs with a ULONGLONG (see winnt.h). This means that the union is only
    // aligned on a 32-bit boundary (for the DWORDs), even though the ULONGLONG
    // requires 64-bit alignment.
    ULARGE_INTEGER m_elapsed;
    ULARGE_INTEGER m_start;

#else // unknown platform

#ifdef STOPWATCH_FINE
    clock_t m_elapsed;
    clock_t m_start;
#else
    time_t m_elapsed;
    time_t m_start;
#endif

#endif

    bool m_running;
    bool m_useWorldTime;

};

#endif
