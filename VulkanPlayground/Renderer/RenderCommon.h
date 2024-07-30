#ifndef RENDERCOMMON_H
#define RENDERCOMMON_H

typedef unsigned char       byte;   // 8 bits
typedef unsigned short      word;   // 16 bits
typedef unsigned int        dword;  // 32 bits
typedef unsigned int        uint;
typedef unsigned long       ulong;

typedef signed char         int8;
typedef unsigned char       uint8;
typedef short int           int16;
typedef unsigned short int  uint16;
typedef int                 int32;
typedef unsigned int        uint32;
typedef long long           int64;
typedef unsigned long long  uint64;

static const int MAX_DESC_SETS              = 16384;
static const int MAX_DESC_UNIFORM_BUFFERS   = 8192;
static const int MAX_DESC_IMAGE_SAMPLERS    = 12384;
static const int MAX_DESC_SET_WRITES        = 32;
static const int MAX_DESC_SET_UNIFORMS      = 48;
static const int MAX_IMAGE_PARMS            = 16;
static const int MAX_UBO_PARMS              = 2;
static const int NUM_TIMESTAMP_QUERIES      = 16;

// vertCacheHandle_t packs size, offset, and frame number into 64 bits
typedef uint64 vertCacheHandle_t;

#endif
