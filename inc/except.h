/*++

Copyright (c) 2023  Dr. Chang Liu, PhD.

Module Name:

    except.h

Abstract:

    This header file contains definitions for all the exception classes thrown by the
    REAPERS library.

Revision History:

    2023-01-05  File created

--*/

#pragma once

#ifndef _REAPERS_H_
#error "Do not include this file directly. Include the master header file reapers.h"
#include <STOP_NOW_AND_FIX_YOUR_DAMN_CODE>
#endif

#define ThrowException(Ex, ...)						\
    throw REAPERS::Ex(__FILE__, __LINE__, __func__, __VA_ARGS__)

#ifdef REAPERS_DEBUG
#define DbgThrow(...)	ThrowException(__VA_ARGS__)
#else
#define DbgThrow(...)
#endif

// Base exception class that records the file name, line number, and function name
// where the exception has happened.
class Exception : public std::exception {
protected:
    std::string msg;
public:
    Exception(const char *file, int line, const char *func) {
	std::stringstream ss;
	ss << file << ":" << line << "@" << func << "(): ";
	msg = ss.str();
    }

    virtual const char *what() const noexcept {
	return msg.c_str();
    }
};

// Exception class thrown when the user tries to construct an object (spin operator
// or field operator) with an invalid index.
class IndexTooLarge : public Exception {
public:
    IndexTooLarge(const char *f, int l, const char *fn, size_t n,
		  size_t nmax, const char *ty) : Exception(f, l, fn) {
	std::stringstream ss;
	ss << "Specified " << ty << " index " << n << " exceeds " << nmax << ".";
	msg += ss.str();
    }
};

class IndexTooSmall : public Exception {
public:
    IndexTooSmall(const char *f, int l, const char *fn, size_t n,
		  size_t nmin, const char *ty) : Exception(f, l, fn) {
	std::stringstream ss;
	ss << "Specified " << ty << " index " << n <<
	    " cannot be less than " << nmin << ".";
	msg += ss.str();
    }
};

class SpinIndexTooLarge : public IndexTooLarge {
public:
    SpinIndexTooLarge(const char *f, int l, const char *fn, size_t n,
		      size_t nmax) : IndexTooLarge(f, l, fn, n, nmax, "spin") {}
};

class FieldIndexTooLarge : public IndexTooLarge {
public:
    FieldIndexTooLarge(const char *f, int l, const char *fn, size_t n,
		       size_t nmax) : IndexTooLarge(f, l, fn, n, nmax, "field") {}
};

class StateIndexTooLarge : public IndexTooLarge {
public:
    StateIndexTooLarge(const char *f, int l, const char *fn, size_t n,
		       size_t nmax) : IndexTooLarge(f, l, fn, n, nmax, "state") {}
};

class SpinIndexTooSmall : public IndexTooSmall {
public:
    SpinIndexTooSmall(const char *f, int l, const char *fn, size_t n,
		      size_t nmin) : IndexTooSmall(f, l, fn, n, nmin, "spin") {}
};

class FieldIndexTooSmall : public IndexTooSmall {
public:
    FieldIndexTooSmall(const char *f, int l, const char *fn, size_t n,
		       size_t nmin) : IndexTooSmall(f, l, fn, n, nmin, "field") {}
};

// Exception class thrown when the user passed in an invalid argument to a function
class InvalidArgument : public Exception {
public:
    InvalidArgument(const char *f, int l, const char *fn,
		    const char *name, const char *param_msg) : Exception(f, l, fn) {
	std::stringstream ss;
	ss << "Invalid argument for " << name << ": "
	   << name << " " << param_msg << ".";
	msg += ss.str();
    }
};

// Exception class thrown when a runtume error has occured
class RuntimeError : public Exception {
public:
    RuntimeError(const char *f, int l, const char *fn,
		 const char *lib, int code) : Exception(f, l, fn) {
	std::stringstream ss;
	ss << lib << " runtime error, code " << code << ".";
	msg += ss.str();
    }
};

// Exception class thrown when a CUDA device memory allocation has failed
class DevOutOfMem : public Exception {
public:
    DevOutOfMem(const char *f, int l, const char *fn, size_t size)
	: Exception(f, l, fn) {
	std::stringstream ss;
	ss << "Failed to allocate " << size << " bytes on CUDA device.";
	msg += ss.str();
    }
};
