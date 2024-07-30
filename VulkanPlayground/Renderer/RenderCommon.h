#ifndef RENDERCOMMON_H
#define RENDERCOMMON_H

static const int MAX_DESC_SETS              = 16384;
static const int MAX_DESC_UNIFORM_BUFFERS   = 8192;
static const int MAX_DESC_IMAGE_SAMPLERS    = 12384;
static const int MAX_DESC_SET_WRITES        = 32;
static const int MAX_DESC_SET_UNIFORMS      = 48;
static const int MAX_IMAGE_PARMS            = 16;
static const int MAX_UBO_PARMS              = 2;
static const int NUM_TIMESTAMP_QUERIES      = 16;

// vertCacheHandle_t packs size, offset, and frame number into 64 bits
typedef uint64_t vertCacheHandle_t;

#endif
