#include "Sciara.h"
#include "cal2DBuffer.h"
#include "cudaUtil.cuh"

/*
  &sciara->simulation->vent                // no allocation, vector<TVent>
  &sciara->simulation->total_emitted_lava  // no allocation, double
  *sciara->substates->Sh                   // done
  *sciara->substates->Sh_next              // done
  *sciara->substates->ST                   // done
  *sciara->substates->ST-next              // done
  *sciara->substates->Sz                   // done
  *sciara->substates->Sz_next              // done
  *sciara->substates->Mf                   // done
  *sciara->substates->Mb                   // done
  *sciara->substates->Mhs                  // done
  *sciara->X->Xi                           // done
  *sciara->X->Xj                           // done
*/

void allocateSubstates(Sciara *sciara)
{
  cudaError_t error = cudaMallocManaged(&(sciara)->substates->Sz, sizeof(double) * sciara->domain->rows * sciara->domain->cols);
  checkReturnedError(error, __LINE__, "Error allocating memory for sciara substates->Sz");
  error = cudaMallocManaged(&(sciara)->substates->Sz_next, sizeof(double) * sciara->domain->rows * sciara->domain->cols);
  checkReturnedError(error, __LINE__, "Error allocating memory for sciara substates->Sz_next");
  error = cudaMallocManaged(&(sciara)->substates->Sh, sizeof(double) * sciara->domain->rows * sciara->domain->cols);
  checkReturnedError(error, __LINE__, "Error allocating memory for sciara substates->Sh");
  error = cudaMallocManaged(&(sciara)->substates->Sh_next, sizeof(double) * sciara->domain->rows * sciara->domain->cols);
  checkReturnedError(error, __LINE__, "Error allocating memory for sciara substates->Sh_next");
  error = cudaMallocManaged(&(sciara)->substates->ST, sizeof(double) * sciara->domain->rows * sciara->domain->cols);
  checkReturnedError(error, __LINE__, "Error allocating memory for sciara substates->ST");
  error = cudaMallocManaged(&(sciara)->substates->ST_next, sizeof(double) * sciara->domain->rows * sciara->domain->cols);
  checkReturnedError(error, __LINE__, "Error allocating memory for sciara substates->ST_next");
  error = cudaMallocManaged(&(sciara)->substates->Mf, sizeof(double) * sciara->domain->rows * sciara->domain->cols * NUMBER_OF_OUTFLOWS);
  checkReturnedError(error, __LINE__, "Error allocating memory for sciara substates->Mf");
  // sciara->substates->Mv       = new (std::nothrow)    int[sciara->domain->rows*sciara->domain->cols];
  error = cudaMallocManaged(&(sciara)->substates->Mb, sizeof(bool) * sciara->domain->rows * sciara->domain->cols);
  checkReturnedError(error, __LINE__, "Error allocating memory for sciara substates->Mb");
  error = cudaMallocManaged(&(sciara)->substates->Mhs, sizeof(double) * sciara->domain->rows * sciara->domain->cols);
  checkReturnedError(error, __LINE__, "Error allocating memory for sciara substates->Mhs");
}

void deallocateSubstates(Sciara *sciara)
{
  cudaError_t error;
  if (sciara->substates->Sz)
  {
    error = cudaFree(sciara->substates->Sz);
    checkReturnedError(error, __LINE__, "Error deallocating memory for sciara substates->Sz");
  }
  if (sciara->substates->Sz_next)
  {
    error = cudaFree(sciara->substates->Sz_next);
    checkReturnedError(error, __LINE__, "Error deallocating memory for sciara substates->Sz_next");
  }
  if (sciara->substates->Sh)
  {
    error = cudaFree(sciara->substates->Sh);
    checkReturnedError(error, __LINE__, "Error deallocating memory for sciara substates->Sh");
  }
  if (sciara->substates->Sh_next)
  {
    error = cudaFree(sciara->substates->Sh_next);
    checkReturnedError(error, __LINE__, "Error deallocating memory for sciara substates->Sh_next");
  }
  if (sciara->substates->ST)
  {
    error = cudaFree(sciara->substates->ST);
    checkReturnedError(error, __LINE__, "Error deallocating memory for sciara substates->ST");
  }
  if (sciara->substates->ST_next)
  {
    error = cudaFree(sciara->substates->ST_next);
    checkReturnedError(error, __LINE__, "Error deallocating memory for sciara substates->ST_next");
  }
  if (sciara->substates->Mf)
  {
    error = cudaFree(sciara->substates->Mf);
    checkReturnedError(error, __LINE__, "Error deallocating memory for sciara substates->Mf");
  }
  /*
  if(sciara->substates->Mv)
  {
    delete[] sciara->substates->Mv;
  }
  */
  if (sciara->substates->Mb)
  {
    error = cudaFree(sciara->substates->Mb);
    checkReturnedError(error, __LINE__, "Error deallocating memory for sciara substates->Mb");
  }
  if (sciara->substates->Mhs)
  {
    error = cudaFree(sciara->substates->Mhs);
    checkReturnedError(error, __LINE__, "Error deallocating memory for sciara substates->Mhs");
  }
}

void evaluatePowerLawParams(double PTvent, double PTsol, double value_sol, double value_vent, double &k1, double &k2)
{
  k2 = (log10(value_vent) - log10(value_sol)) / (PTvent - PTsol);
  k1 = log10(value_sol) - k2 * (PTsol);
}

void simulationInitialize(Sciara *sciara)
{
  // dichiarazioni
  unsigned int maximum_number_of_emissions = 0;

  // azzeramento dello step dell'AC
  sciara->simulation->step = 0;
  sciara->simulation->elapsed_time = 0;

  // determinazione numero massimo di passi
  for (unsigned int i = 0; i < sciara->simulation->emission_rate.size(); i++)
    if (maximum_number_of_emissions < sciara->simulation->emission_rate[i].size())
      maximum_number_of_emissions = sciara->simulation->emission_rate[i].size();
  // maximum_steps_from_emissions = (int)(emission_time/Pclock*maximum_number_of_emissions);
  sciara->simulation->effusion_duration = sciara->simulation->emission_time * maximum_number_of_emissions;
  sciara->simulation->total_emitted_lava = 0;

  // definisce il bordo della morfologia
  MakeBorder(sciara);

  // calcolo a b (parametri viscosità) c d (parametri resistenza al taglio)
  evaluatePowerLawParams(
      sciara->parameters->PTvent,
      sciara->parameters->PTsol,
      sciara->parameters->Pr_Tsol,
      sciara->parameters->Pr_Tvent,
      sciara->parameters->a,
      sciara->parameters->b);
  evaluatePowerLawParams(
      sciara->parameters->PTvent,
      sciara->parameters->PTsol,
      sciara->parameters->Phc_Tsol,
      sciara->parameters->Phc_Tvent,
      sciara->parameters->c,
      sciara->parameters->d);
}

int _Xi[] = {0, -1, 0, 0, 1, -1, 1, 1, -1}; // Xj: Moore neighborhood row coordinates (see below)
int _Xj[] = {0, 0, -1, 1, 0, -1, -1, 1, 1}; // Xj: Moore neighborhood col coordinates (see below)
void init(Sciara *&sciara)
{
  sciara = new Sciara;

  (sciara)->domain = new Domain;

  (sciara)->X = new NeighsRelativeCoords;

  cudaError_t error = cudaMallocManaged(&(sciara)->X->Xi, sizeof(int) * MOORE_NEIGHBORS);
  checkReturnedError(error, __LINE__);

  error = cudaMallocManaged(&(sciara)->X->Xj, sizeof(int) * MOORE_NEIGHBORS);
  checkReturnedError(error, __LINE__);

  for (int n = 0; n < MOORE_NEIGHBORS; n++)
  {
    (sciara)->X->Xi[n] = _Xi[n];
    (sciara)->X->Xj[n] = _Xj[n];
  }

  (sciara)->substates = new Substates;
  // allocateSubstates(sciara); // substates allocation is done when the confiugration is loaded (see io.cu)
  (sciara)->parameters = new Parameters;
  (sciara)->simulation = new Simulation;
  error = cudaMallocManaged(&(sciara)->simulation->vent, sizeof(vector<TVent>));
  checkReturnedError(error, __LINE__, "Error allocating memory for sciara simulation->vent");
}

void finalize(Sciara *&sciara)
{
  deallocateSubstates((sciara));
  delete (sciara)->domain;
  cudaFree(sciara->X->Xi);
  cudaFree(sciara->X->Xj);
  delete (sciara)->X;
  delete (sciara)->substates;
  delete (sciara)->parameters;
  delete (sciara)->simulation;
  delete sciara;
  sciara = NULL;
}

void MakeBorder(Sciara *sciara)
{
  int j, i;

  // prima riga
  i = 0;
  for (j = 0; j < sciara->domain->cols; j++)
    if (calGetMatrixElement(sciara->substates->Sz, sciara->domain->cols, i, j) >= 0)
      calSetMatrixElement(sciara->substates->Mb, sciara->domain->cols, i, j, true);

  // ultima riga
  i = sciara->domain->rows - 1;
  for (j = 0; j < sciara->domain->cols; j++)
    if (calGetMatrixElement(sciara->substates->Sz, sciara->domain->cols, i, j) >= 0)
      calSetMatrixElement(sciara->substates->Mb, sciara->domain->cols, i, j, true);

  // prima colonna
  j = 0;
  for (i = 0; i < sciara->domain->rows; i++)
    if (calGetMatrixElement(sciara->substates->Sz, sciara->domain->cols, i, j) >= 0)
      calSetMatrixElement(sciara->substates->Mb, sciara->domain->cols, i, j, true);

  // ultima colonna
  j = sciara->domain->cols - 1;
  for (i = 0; i < sciara->domain->rows; i++)
    if (calGetMatrixElement(sciara->substates->Sz, sciara->domain->cols, i, j) >= 0)
      calSetMatrixElement(sciara->substates->Mb, sciara->domain->cols, i, j, true);

  // il resto
  for (int i = 1; i < sciara->domain->rows - 1; i++)
    for (int j = 1; j < sciara->domain->cols - 1; j++)
      if (calGetMatrixElement(sciara->substates->Sz, sciara->domain->cols, i, j) >= 0)
      {
        for (int k = 1; k < MOORE_NEIGHBORS; k++)
          if (calGetMatrixElement(sciara->substates->Sz, sciara->domain->cols, i + sciara->X->Xi[k], j + sciara->X->Xj[k]) < 0)
          {
            calSetMatrixElement(sciara->substates->Mb, sciara->domain->cols, i, j, true);
            break;
          }
      }
}
