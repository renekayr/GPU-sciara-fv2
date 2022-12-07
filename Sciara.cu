#include "Sciara.h"
#include "cal2DBuffer.h"
#include "cudaUtil.cuh"

void allocateSubstates(Sciara *sciara)
{
  cudaError_t error = cudaMallocManaged((void**)&sciara -> substates -> Sz, sizeof(double[sciara -> domain -> rows * sciara -> domain -> cols]));
  checkReturnedError(error, __LINE__);
  error = cudaMallocManaged((void**)&sciara -> substates -> Sz_next, sizeof(double[sciara -> domain -> rows * sciara -> domain -> cols]));
  checkReturnedError(error, __LINE__);
  error = cudaMallocManaged((void**)&sciara -> substates -> Sh, sizeof(double[sciara -> domain -> rows * sciara -> domain -> cols]));
  checkReturnedError(error, __LINE__);
  error = cudaMallocManaged((void**)&sciara -> substates -> Sh_next, sizeof(double[sciara -> domain -> rows * sciara -> domain -> cols]));
  checkReturnedError(error, __LINE__);
  error = cudaMallocManaged((void**)&sciara -> substates -> ST, sizeof(double[sciara -> domain -> rows * sciara -> domain -> cols]));
  checkReturnedError(error, __LINE__);
  error = cudaMallocManaged((void**)&sciara -> substates -> ST_next, sizeof(double[sciara -> domain -> rows * sciara -> domain -> cols]));
  checkReturnedError(error, __LINE__);
  error = cudaMallocManaged((void**)&sciara -> substates -> Mf, sizeof(double[sciara -> domain -> rows * sciara -> domain -> cols * NUMBER_OF_OUTFLOWS]));
  checkReturnedError(error, __LINE__);
  // sciara->substates->Mv       = new (std::nothrow)    int[sciara->domain->rows*sciara->domain->cols];
  error = cudaMallocManaged((void**)&sciara -> substates -> Mb, sizeof(bool[sciara -> domain -> rows * sciara -> domain -> cols]));
  checkReturnedError(error, __LINE__);
  error = cudaMallocManaged((void**)&sciara -> substates -> Mhs, sizeof(double[sciara -> domain -> rows * sciara -> domain -> cols]));
  checkReturnedError(error, __LINE__);
}

void deallocateSubstates(Sciara *sciara)
{
  cudaError_t error;
  if (sciara->substates->Sz)
    error = cudaFree(sciara->substates->Sz);
  if (sciara->substates->Sz_next)
    error = cudaFree(sciara->substates->Sz_next);
  if (sciara->substates->Sh)
    error = cudaFree(sciara->substates->Sh);
  if (sciara->substates->Sh_next)
    error = cudaFree(sciara->substates->Sh_next);
  if (sciara->substates->ST)
    error = cudaFree(sciara->substates->ST);
  if (sciara->substates->ST_next)
    error = cudaFree(sciara->substates->ST_next);
  if (sciara->substates->Mf)
    error = cudaFree(sciara->substates->Mf);
  // if(sciara->substates->Mv)       cudaFree(sciara->substates->Mv);
  if (sciara->substates->Mb)
    error = cudaFree(sciara->substates->Mb);
  if (sciara->substates->Mhs)
    error = cudaFree(sciara->substates->Mhs);
  
  checkReturnedError(error, __LINE__);
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
  cudaError_t error = cudaMallocManaged((void**)&sciara, sizeof(Sciara));
  checkReturnedError(error, __LINE__);
  error = cudaMallocManaged((void**)&sciara -> domain, sizeof(Domain));
  checkReturnedError(error, __LINE__);

  error = cudaMallocManaged((void**)&sciara -> X, sizeof(NeighsRelativeCoords));
  checkReturnedError(error, __LINE__);
  error = cudaMallocManaged((void**)&sciara -> X -> Xi, sizeof(int[MOORE_NEIGHBORS]));
  checkReturnedError(error, __LINE__);
  error = cudaMallocManaged((void**)&sciara -> X -> Xj, sizeof(int[MOORE_NEIGHBORS]));
  checkReturnedError(error, __LINE__);
  
  for (int n = 0; n < MOORE_NEIGHBORS; n++)
  {
    sciara->X->Xi[n] = _Xi[n];
    sciara->X->Xj[n] = _Xj[n];
  }

  error = cudaMallocManaged((void**)&sciara -> substates, sizeof(Substates));
  checkReturnedError(error, __LINE__);
  // allocateSubstates(sciara); //Substates allocation is done when the confiugration is loaded
  error = cudaMallocManaged((void**)&sciara -> parameters, sizeof(Parameters));
  checkReturnedError(error, __LINE__);
  error = cudaMallocManaged((void**)&sciara -> simulation, sizeof(Simulation));
  checkReturnedError(error, __LINE__);
}

void finalize(Sciara *&sciara)
{
  deallocateSubstates(sciara);
  cudaError_t error = cudaFree(sciara->domain);
  checkReturnedError(error, __LINE__);
  cudaFree(sciara->X->Xi);
  checkReturnedError(error, __LINE__);
  cudaFree(sciara->X->Xj);
  checkReturnedError(error, __LINE__);
  cudaFree(sciara->X);
  checkReturnedError(error, __LINE__);
  cudaFree(sciara->substates);
  checkReturnedError(error, __LINE__);
  cudaFree(sciara->parameters);
  checkReturnedError(error, __LINE__);
  cudaFree(sciara->simulation);
  checkReturnedError(error, __LINE__);
  cudaFree(sciara);
  checkReturnedError(error, __LINE__);
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
