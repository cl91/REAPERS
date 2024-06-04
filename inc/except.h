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
    throw REAPERS::Ex(__FILE__, __LINE__, __func__, ##__VA_ARGS__)

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

    InvalidArgument(const char *f, int l, const char *fn,
		    const char *err_msg) : Exception(f, l, fn) {
	msg += std::string("Invalid argument: ") + err_msg;
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

#ifndef REAPERS_NOGPU
// Exception class thrown when a runtume error has occured
class CudaError : public Exception {
public:
    CudaError(const char *f, int l, const char *fn, cudaError_t code) : Exception(f, l, fn) {
	std::stringstream ss;
	ss << " GPU runtime error, code " << code << " ("
	   << cudaGetErrorString(code) << ").";
	msg += ss.str();
    }
};
#endif

// Exception class thrown when there is no GPU device available
class NoGpuDevice : public Exception {
public:
    NoGpuDevice(const char *f, int l, const char *fn)
	: Exception(f, l, fn) {
	msg += "No GPU device available";
    }
};

class NoP2PAccess : public Exception {
public:
    NoP2PAccess(const char *f, int l, const char *fn, int dev0, int dev1)
	: Exception(f, l, fn) {
	std::stringstream ss;
	ss << "P2P access between device " << dev0 << " and device " << dev1
	   << " is not supported.";
	msg += ss.str();
    }
};

// Exception class thrown when a GPU device memory allocation has failed
class DevOutOfMem : public Exception {
public:
    DevOutOfMem(const char *f, int l, const char *fn, size_t size)
	: Exception(f, l, fn) {
	std::stringstream ss;
	ss << "Failed to allocate " << size << " bytes on GPU device.";
	msg += ss.str();
    }
};
