// ============================================================================
// gzstream, C++ iostream classes wrapping the zlib compression library.
// Copyright (C) 2001  Deepak Bandyopadhyay, Lutz Kettner
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
// ============================================================================
//
// File          : gzstream.h
// Revision      : $Revision: 1.5 $
// Revision_date : $Date: 2002/04/26 23:30:15 $
// Author(s)     : Deepak Bandyopadhyay, Lutz Kettner
//
// Standard streambuf implementation following Nicolai Josuttis, "The
// Standard C++ Library".
// ============================================================================

#ifndef GZSTREAM_H
#define GZSTREAM_H 1


#include <fstream>
#include <iostream>
#include <zlib.h>

#ifdef GZSTREAM_NAMESPACE
namespace GZSTREAM_NAMESPACE {
#endif



class gzstreambuf : public std::streambuf {
private:
  static const int bufferSize = 47 + 256; 

  gzFile file;            
  char buffer[bufferSize]; 
  char opened;            
  int mode;               

  int flush_buffer();

public:
  gzstreambuf() : opened(0) {
    setp(buffer, buffer + (bufferSize - 1));
    setg(buffer + 4,  
         buffer + 4,  
         buffer + 4); 
    
  }
  int is_open() { return opened; }
  gzstreambuf *open(const char *name, int open_mode);
  gzstreambuf *close();
  ~gzstreambuf() { close(); }

  virtual int overflow(int c = EOF);
  virtual int underflow();
  virtual int sync();
};

class gzstreambase : virtual public std::ios {
protected:
  gzstreambuf buf;

public:
  gzstreambase() { init(&buf); }
  gzstreambase(const char *name, int open_mode);
  ~gzstreambase();
  void open(const char *name, int open_mode);
  void close();
  gzstreambuf *rdbuf() { return &buf; }
};


class igzstream : public gzstreambase, public std::istream {
public:
  igzstream() : std::istream(&buf) {}
  igzstream(const char *name, int open_mode = std::ios::in)
      : gzstreambase(name, open_mode), std::istream(&buf) {}
  gzstreambuf *rdbuf() { return gzstreambase::rdbuf(); }
  void open(const char *name, int open_mode = std::ios::in) {
    gzstreambase::open(name, open_mode);
  }
};

class ogzstream : public gzstreambase, public std::ostream {
public:
  ogzstream() : std::ostream(&buf) {}
  ogzstream(const char *name, int mode = std::ios::out)
      : gzstreambase(name, mode), std::ostream(&buf) {}
  gzstreambuf *rdbuf() { return gzstreambase::rdbuf(); }
  void open(const char *name, int open_mode = std::ios::out) {
    gzstreambase::open(name, open_mode);
  }
};

#ifdef GZSTREAM_NAMESPACE
} 
#endif

#endif 

