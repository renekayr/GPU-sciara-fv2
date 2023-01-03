#include "cal2DBuffer.h"
#include "configurationPathLib.h"
#include "GISInfo.h"
#include "io.h"
#include "vent.h"
#include <omp.h>
#include <new>
#include "Sciara.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "util.hpp"
#include "cudaUtil.cuh"

#define gpuErrchk(ans)                    \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
  }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

// ----------------------------------------------------------------------------
// I/O parameters used to index argv[]
// ----------------------------------------------------------------------------

#define INPUT_PATH_ID 1
#define OUTPUT_PATH_ID 2
#define MAX_STEPS_ID 3
#define REDUCE_INTERVL_ID 4
#define THICKNESS_THRESHOLD_ID 5

// ----------------------------------------------------------------------------
// Read/Write access macros linearizing single/multy layer buffer 2D indices
// ----------------------------------------------------------------------------

#define SET(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define GET(M, columns, i, j) (M[(((i) * (columns)) + (j))])
#define BUF_SET(M, rows, columns, n, i, j, value) ((M)[(((n) * (rows) * (columns)) + ((i) * (columns)) + (j))] = (value))
#define BUF_GET(M, rows, columns, n, i, j) (M[(((n) * (rows) * (columns)) + ((i) * (columns)) + (j))])

// ----------------------------------------------------------------------------
// Computing kernels, aka elementary processes in the XCA terminology
// ----------------------------------------------------------------------------

// No tiled implementation as each thread only accesses one element
__global__ void emitLavaKernel(
    int r,
    int c,
    TVent *vent,
    int n_vent,
    double elapsed_time,
    double Pclock,
    double emission_time,
    double &total_emitted_lava,
    double Pac,
    double PTvent,
    double *Sh,
    double *Sh_next,
    double *ST_next)
{
  long row_index = threadIdx.y + blockDim.y * blockIdx.y;
  long col_index = threadIdx.x + blockDim.x * blockIdx.x;
  long row_stride = blockDim.y * gridDim.y;
  long col_stride = blockDim.x * gridDim.x;

  for (long row = row_index; row < r; row += row_stride)
  {
    for (long col = col_index; col < c; col += col_stride)
    {
      for (int k = 0; k < n_vent; ++k)
      {
        if (row == vent[k].y() && col == vent[k].x())
        {
          SET(Sh_next, c, row, col, GET(Sh, c, row, col) + vent[k].thickness(elapsed_time, Pclock, emission_time, Pac));
          SET(ST_next, c, row, col, PTvent);

          total_emitted_lava += vent[k].thickness(elapsed_time, Pclock, emission_time, Pac);
        }
      }
    }
  }
}

// This kernel utilizes a tiled algorithm with halo cells
__global__ void computeOutflowsKernel(
    int r,
    int c,
    int *Xi,
    int *Xj,
    double *Sz,
    double *Sh,
    double *ST,
    double *Mf,
    double Pc,
    double _a,
    double _b,
    double _c,
    double _d)
{
  long row_index = threadIdx.y + blockDim.y * blockIdx.y;
  long col_index = threadIdx.x + blockDim.x * blockIdx.x;
  long row_stride = blockDim.y * gridDim.y;
  long col_stride = blockDim.x * gridDim.x;

  // TODO: Implement tiled algorithm with halo cells
  //  - determine what to buffer in shared memory
  //  - determine how to access the buffers correctly (indexing)
  //  - 

  bool eliminated[MOORE_NEIGHBORS];
  double z[MOORE_NEIGHBORS];
  double h[MOORE_NEIGHBORS];
  double H[MOORE_NEIGHBORS];
  double theta[MOORE_NEIGHBORS];
  double w[MOORE_NEIGHBORS];  // Distances between central and adjecent cells
  double Pr[MOORE_NEIGHBORS]; // Relaiation rate array
  bool loop;
  int counter;
  double sz0, sz, T, avg, rr, hc;

  for (long row = row_index; row < r; row += row_stride)
  {
    for (long col = col_index; col < c; col += col_stride)
    {
      printf("computeOutflows: (%d, %d)\n", row, col);
      if (GET(Sh, c, row, col) <= 0)
        return;

      T = GET(ST, c, row, col);
      rr = pow(10, _a + _b * T);
      hc = pow(10, _c + _d * T);

      for (int k = 0; k < MOORE_NEIGHBORS; ++k)
      {
        sz0 = GET(Sz, c, row, col);
        sz = GET(Sz, c, row + Xi[k], col + Xj[k]);
        h[k] = GET(Sh, c, row + Xi[k], col + Xj[k]);
        w[k] = Pc;
        Pr[k] = rr;

        if (k < VON_NEUMANN_NEIGHBORS)
          z[k] = sz;
        else
          z[k] = sz0 - (sz0 - sz) / sqrt(2.0);
      }

      H[0] = z[0];
      theta[0] = 0;
      eliminated[0] = false;
      for (int k = 1; k < MOORE_NEIGHBORS; ++k)
      {
        if (z[0] + h[0] > z[k] + h[k])
        {
          H[k] = z[k] + h[k];
          theta[k] = atan(((z[0] + h[0]) - (z[k] + h[k])) / w[k]);
          eliminated[k] = false;
        }
        else
        {
          // H[k] = 0;
          // theta[k] = 0;
          eliminated[k] = true;
        }
      }

      do
      {
        loop = false;
        avg = h[0];
        counter = 0;
        for (int k = 0; k < MOORE_NEIGHBORS; ++k)
          if (!eliminated[k])
          {
            avg += H[k];
            ++counter;
          }
        if (counter != 0)
          avg = avg / double(counter);
        for (int k = 0; k < MOORE_NEIGHBORS; ++k)
          if (!eliminated[k] && avg <= H[k])
          {
            eliminated[k] = true;
            loop = true;
          }
      } while (loop);

      for (int k = 1; k < MOORE_NEIGHBORS; ++k)
      {
        if (!eliminated[k] && h[0] > hc * cos(theta[k]))
          BUF_SET(Mf, r, c, k - 1, row, col, Pr[k] * (avg - H[k]));
        else
          BUF_SET(Mf, r, c, k - 1, row, col, 0.0);
      }
    }
  }
}

// This kernel utilizes a tiled algorithm with halo cells
__global__ void massBalanceKernel(
    int r,
    int c,
    int *Xi,
    int *Xj,
    double *Sh,
    double *Sh_next,
    double *ST,
    double *ST_next,
    double *Mf,
    int tile_size,
    int mask_size)
{
  long col_index = threadIdx.x + tile_size * blockIdx.x;
  long row_index = threadIdx.y + tile_size * blockIdx.y;
  long col_halo = col_index - mask_size/2;
  long row_halo = row_index - mask_size/2;

  __constant__ const int inflowsIndices[NUMBER_OF_OUTFLOWS] = {3, 2, 1, 0, 6, 7, 4, 5};
  double inFlow, outFlow, neigh_t, initial_h, initial_t, h_next, t_next;

  // With 2 buffers, total consumed shared memory in bytes is
  // 2 * sizeof(double) * (tile_size + mask_size - 1)^2
  __shared__ double ST_ds[pow(tile_size + mask_size - 1, 2)];
  __shared__ double Mf_ds[8 * pow(tile_size + mask_size - 1, 2)];

  // !TODO: replace row and col (leftover from grid stride) with appropriate values

  // Phase 1: All block threads copy values into shared memory
  // TODO: Individual check for each buffer because they are indexed differently
  if((col_halo >= 0) && (col_halo < c) && (row_halo >= 0) && (row_halo < r)) {
    ST_ds[threadIdx.x + threadIdx.y * c] = GET(ST, c, row + Xi[n], col + Xj[n]);  // TODO: load appropriate values into shared memory
    for(int n = 1; n < MOORE_NEIGHBORS; ++n) {
      // TODO: load appropriate values into shared memory
      // TODO: offset populating indices of Mf according to tile to have indices match up
      // (indexing assumes all of MF, but buffer is smaller)
      Mf_ds[((inflowsIndices[n - 1]) * (r) * (c)) + ((row + Xi[n]) * (c)) + (col + Xj[n])] = 
        BUF_GET(Mf_ds, r, c, inflowsIndices[n - 1], row + Xi[n], col + Xj[n]);
    }
  }
  // TODO: Determine what happens in the serial implementation at the edge of domain,
  // e.g. when accessing moore neighbors of ST[0,0] and determine how to handle these accesses with haloes.
  // Insert neutral element or not treat this case at all?
  //
  // else {
  //   ST_ds[threadIdx.x + threadIdx.y * c] = 0.0;  // TODO: Check how serial implementation handles out of bounds indexing?
  //   Mf_ds[] = 0.0;  // TODO: Check how serial implementation handles out of bounds indexing?
  // }
  __syncthreads();

  // phase 2: tile threads compute outputs (no grid stride)
  if(threadIdx.y < tile_size && threadIdx.y < tile_size) {
    initial_h = GET(Sh, c, row, col);
    initial_t = GET(ST, c, row, col);
    h_next = initial_h;
    t_next = initial_h * initial_t;

    for (int n = 1; n < MOORE_NEIGHBORS; ++n)
    {
      neigh_t = GET(ST_ds, c, row + Xi[n], col + Xj[n]);  // TODO: Access shared memory with correct index
      // inFlow = BUF_GET(Mf_ds, r, c, inflowsIndices[n - 1], row + Xi[n], col + Xj[n]);  // TODO: Access shared memory with correct index
      // outFlow = BUF_GET(Mf_ds, r, c, n - 1, row, col);  // TODO: Access shared memory with correct index

      h_next += inFlow - outFlow;
      t_next += (inFlow * neigh_t - outFlow * initial_t);
    }

    if (h_next > 0)
    {
      t_next /= h_next;
      SET(ST_next, c, row, col, t_next);
      SET(Sh_next, c, row, col, h_next);
    }
  }
}

// No tiled implementation. There are no overlapping buffer accesses -> no possible performance gain
__global__ void computeNewTemperatureAndSolidificationKernel(
    int r,
    int c,
    double Pepsilon,
    double Psigma,
    double Pclock,
    double Pcool,
    double Prho,
    double Pcv,
    double Pac,
    double PTsol,
    double *Sz,
    double *Sz_next,
    double *Sh,
    double *Sh_next,
    double *ST,
    double *ST_next,
    double *Mf,
    double *Mhs,
    bool *Mb)
{
  long col_index = threadIdx.x + blockDim.x * blockIdx.x;
  long row_index = threadIdx.y + blockDim.y * blockIdx.y;
  long row_stride = blockDim.y * gridDim.y;
  long col_stride = blockDim.x * gridDim.x;

  double nT, aus, z, h, T;

  for (long row = row_index; row < r; row += row_stride)
  {
    for (long col = col_index; col < c; col += col_stride)
    {
      z = GET(Sz, c, row, col);
      h = GET(Sh, c, row, col);
      T = GET(ST, c, row, col);

      if (h > 0 && GET(Mb, c, row, col) == false)
      {
        aus = 1.0 + (3 * pow(T, 3.0) * Pepsilon * Psigma * Pclock * Pcool) / (Prho * Pcv * h * Pac);
        nT = T / pow(aus, 1.0 / 3.0);

        if (nT > PTsol) // no solidification
          SET(ST_next, c, row, col, nT);
        else // solidification
        {
          SET(Sz_next, c, row, col, z + h);
          SET(Sh_next, c, row, col, 0.0);
          SET(ST_next, c, row, col, PTsol);
          SET(Mhs, c, row, col, GET(Mhs, c, row, col) + h);
        }
      }
    }
  }
}

void boundaryConditions(
    int i, int j,
    int r,
    int c,
    double *Mf,
    bool *Mb,
    double *Sh,
    double *Sh_next,
    double *ST,
    double *ST_next)
{
  return;  // what?
  if (GET(Mb, c, i, j))
  {
    SET(Sh_next, c, i, j, 0.0);
    SET(ST_next, c, i, j, 0.0);
  }
}

double reduceAdd(int r, int c, double *buffer)
{
  double sum = 0.0;
  for (int i = 0; i < r; ++i)
    for (int j = 0; j < c; ++j)
      sum += GET(buffer, c, i, j);

  return sum;
}

// ----------------------------------------------------------------------------
// Function main()
// ----------------------------------------------------------------------------
int main(int argc, char **argv)
{
  Sciara *sciara;
  init(sciara);

  // Input data
  int max_steps = atoi(argv[MAX_STEPS_ID]);
  loadConfiguration(argv[INPUT_PATH_ID], sciara);

  // Domain boundaries and neighborhood
  int i_start = 0, i_end = sciara->domain->rows; // [i_start,i_end[: kernels application range along the rows
  int j_start = 0, j_end = sciara->domain->cols; // [j_start,j_end[: kernels application range along the cols

  // Simulation initialization and loop
  double total_current_lava = -1;
  simulationInitialize(sciara);
  util::Timer cl_timer;
  int reduceInterval = atoi(argv[REDUCE_INTERVL_ID]);
  double thickness_threshold = atof(argv[THICKNESS_THRESHOLD_ID]);

  long n = (i_end - 1) * (j_end - 1);  // problem size

  // CUDA init
  int non_tiled_block_size_x = 4;
  int non_tiled_block_size_y = 4;
  dim3 non_tiled_block_size(dim_x, dim_y, 1);
  dim3 non_tiled_grid_size(ceil(n / non_tiled_block_size_x), ceil(n / non_tiled_block_size_y), 1);

  // Tile size can be dynamically calculated by using tile_size = 1 - mask_width - (pow(max_shared_memory, 2) / pow(sizeof(datatype), 2))
  // max_shared_memory can be queried from the CUDA API at runtime
  // This formula is derived by solving the following equation for for tile_size:
  // max_shared_memory = (mask_size + tile_size - 1)^2 * sizeof(datatype)
  int tile_size = 4;  // else, an arbitrary or estimated amount that does not surpass the GPU's capacity is chosen
  int block_width = tile_size + mask_size - 1;
  dim3 block_size(block_width, block_width, 1);
  dim3 grid_size(ceil(n / tile_size), ceil(n / tile_size), 1);
  // Apply the emitLava kernel to the whole domain and update the Sh and ST state variables
  // For &(*sciara->simulation->vent)[0], assume the STL-vector specification to guarantee contiguous storage of elements (http://www.open-std.org/jtc1/sc22/wg21/docs/lwg-defects.html#69)
  emitLavaKernel<<<non_tiled_grid_size, non_tiled_block_size>>>(
      sciara->domain->rows,
      sciara->domain->cols,
      &(*sciara->simulation->vent)[0],
      (*sciara->simulation->vent).size(),
      sciara->simulation->elapsed_time,
      sciara->parameters->Pclock,
      sciara->simulation->emission_time,
      sciara->simulation->total_emitted_lava,
      sciara->parameters->Pac,
      sciara->parameters->PTvent,
      sciara->substates->Sh,
      sciara->substates->Sh_next,
      sciara->substates->ST_next);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  memcpy(sciara->substates->Sh, sciara->substates->Sh_next, sizeof(double) * sciara->domain->rows * sciara->domain->cols);
  memcpy(sciara->substates->ST, sciara->substates->ST_next, sizeof(double) * sciara->domain->rows * sciara->domain->cols);

  //  Apply the computeOutflows kernel to the whole domain
  // TODO: Adapt kernel launch to tiled grid
  computeOutflowsKernel<<<grid_size, block_size>>>(
      sciara->domain->rows,
      sciara->domain->cols,
      sciara->X->Xi,
      sciara->X->Xj,
      sciara->substates->Sz,
      sciara->substates->Sh,
      sciara->substates->ST,
      sciara->substates->Mf,
      sciara->parameters->Pc,
      sciara->parameters->a,
      sciara->parameters->b,
      sciara->parameters->c,
      sciara->parameters->d);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  // Apply the massBalance mass balance kernel to the whole domain and update the Sh and ST state variables
  massBalanceKernel<<<grid_size, block_size>>>(sciara->domain->rows,
                                               sciara->domain->cols,
                                               sciara->X->Xi,
                                               sciara->X->Xj,
                                               sciara->substates->Sh,
                                               sciara->substates->Sh_next,
                                               sciara->substates->ST,
                                               sciara->substates->ST_next,
                                               sciara->substates->Mf);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  memcpy(sciara->substates->Sh, sciara->substates->Sh_next, sizeof(double) * sciara->domain->rows * sciara->domain->cols);
  memcpy(sciara->substates->ST, sciara->substates->ST_next, sizeof(double) * sciara->domain->rows * sciara->domain->cols);

  // Apply the computeNewTemperatureAndSolidification kernel to the whole domain
  computeNewTemperatureAndSolidificationKernel<<<non_tiled_grid_size, non_tiled_block_size>>>(
      sciara->domain->rows,
      sciara->domain->cols,
      sciara->parameters->Pepsilon,
      sciara->parameters->Psigma,
      sciara->parameters->Pclock,
      sciara->parameters->Pcool,
      sciara->parameters->Prho,
      sciara->parameters->Pcv,
      sciara->parameters->Pac,
      sciara->parameters->PTsol,
      sciara->substates->Sz,
      sciara->substates->Sz_next,
      sciara->substates->Sh,
      sciara->substates->Sh_next,
      sciara->substates->ST,
      sciara->substates->ST_next,
      sciara->substates->Mf,
      sciara->substates->Mhs,
      sciara->substates->Mb);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  memcpy(sciara->substates->Sz, sciara->substates->Sz_next, sizeof(double) * sciara->domain->rows * sciara->domain->cols);
  memcpy(sciara->substates->Sh, sciara->substates->Sh_next, sizeof(double) * sciara->domain->rows * sciara->domain->cols);
  memcpy(sciara->substates->ST, sciara->substates->ST_next, sizeof(double) * sciara->domain->rows * sciara->domain->cols);

  // Apply the boundaryConditions kernel to the whole domain and update the Sh and ST state variables
  #pragma omp parallel for
    for (int i = i_start; i < i_end; i++)
      for (int j = j_start; j < j_end; j++)
        boundaryConditions(i, j,
                           sciara->domain->rows,
                           sciara->domain->cols,
                           sciara->substates->Mf,
                           sciara->substates->Mb,
                           sciara->substates->Sh,
                           sciara->substates->Sh_next,
                           sciara->substates->ST,
                           sciara->substates->ST_next);
    memcpy(sciara->substates->Sh, sciara->substates->Sh_next, sizeof(double) * sciara->domain->rows * sciara->domain->cols);
    memcpy(sciara->substates->ST, sciara->substates->ST_next, sizeof(double) * sciara->domain->rows * sciara->domain->cols);

    // Global reduction
    if (sciara->simulation->step % reduceInterval == 0)
      total_current_lava = reduceAdd(sciara->domain->rows, sciara->domain->cols, sciara->substates->Sh);

    double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
    printf("Elapsed time [s]: %lf\n", cl_time);
    printf("Emitted lava [m]: %lf\n", sciara->simulation->total_emitted_lava);
    printf("Current lava [m]: %lf\n", total_current_lava);

    printf("Saving output to %s...\n", argv[OUTPUT_PATH_ID]);
    saveConfiguration(argv[OUTPUT_PATH_ID], sciara);

  printf("Releasing memory...\n");
  finalize(sciara);

  return 0;
}
