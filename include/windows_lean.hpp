// WINDOWS_LEAN.hpp - MUST be included PRIOR to any Boost or Windows headers.
// Omitting these macros causes <boost/beast.hpp> to transitively corrupt our
// DSL token structure: Windows GDI headers (wingdi.h) define `type` as a macro,
// effectively destroying the `TokenType type;` member definition.
#pragma once

// Excludes rarely-used APIs (GDI, RPC, etc.), significantly reducing header
// bloat (~5MB) and preventing catastrophic macro collisions.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

// Prevents <windows.h> from defining min/max macros, which would otherwise
// shadow and break std::min/std::max from <algorithm>.
#ifndef NOMINMAX
#define NOMINMAX
#endif

// <winsock.h> and <winsock2.h> are mutually incompatible. Boost.Asio/Beast
// explicitly require winsock2. If <windows.h> is included previously without
// _WINSOCK2_, compilation halts. Defined preemptively here.
#ifndef _WINSOCK2_
#define _WINSOCK2_
#endif
