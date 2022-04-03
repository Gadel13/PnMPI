/* \brief Count MPI func calls time.
 *
 * \details This module counts how often each MPI call was invoked. Before MPI
 *  finalizes, statistics will be printed for each rank and a sum of the made
 *  calls by all ranks.
 */


#include <stdio.h>
#include <stddef.h>
#include <sys/time.h>

#include <mpi.h>
#include <pnmpi/debug_io.h>
#include <pnmpi/hooks.h>
#include <pnmpi/private/pmpi_assert.h>
#include <pnmpi/xmpi.h>

#include "pnmpi-atomic.h"



typedef unsigned long long timing_t;

timing_t get_time_ns() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000000000ll + tv.tv_usec * 1000ll;
}


/** \brief Helper function to start and end \p timer.
 *
 *
 * \param timer The timer to be started / stoped.
 *
 * \return 0 The timer was started.
 * \return >0 The timer was stoped and the elapsed time will be returned.
 */
static timing_t start_stop_timer(timing_t *t)
{
  if (*t != 0)
    {
      timing_t tmp = get_time_ns() - *t;
      *t = 0;
      return tmp;
    }

  *t = get_time_ns();
  return 0;
}


/* If there is no atomic support, we'll limit the threading level of this module
 * to MPI_THREAD_SERIALIZED, so it is safe to use with threaded applications
 * (but they may become slower!). */
#if defined(METRIC_NO_ATOMIC)
const int PNMPI_SupportedThreadingLevel = MPI_THREAD_SERIALIZED;
#endif


/** \brief Struct of counters.
 *
 * \details This struct stores the counters for all MPI calls.
 */
static struct counter
{
  {{forallfn fn_name MPI_Finalize}}
  metric_atomic_keyword size_t {{fn_name}};
  {{endforallfn}}
} counters = { 0 };


/** \brief Helper function to print counter struct.
 *
 * \details This function will print all functions called at least once with
 *  their counter.
 *
 *
 * \param c \ref counter struct to be printed.
 */
static void print_counters(struct counter *c)
{
  {{forallfn fn_name MPI_Finalize}}
  if (c->{{fn_name}} > 0)
    printf("  %8zu %s\n", c->{{fn_name}}, "{{fn_name}}");
  {{endforallfn}}
}


/* Generate wrapper functions for all MPI calls to increment the counter on
 * every call. MPI_Finalize will be ignored, because it will be used to print
 * the statistics below. */

{{fnall fn_name MPI_Finalize MPI_Pcontrol}}
    
  metric_atomic_add(counters.{{fn_name}}, start_stop_timer(&timer));
  WRAP_MPI_CALL_PREFIX
  X{{fn_name}}({{args}});
  WRAP_MPI_CALL_POSTFIX
  metric_atomic_add(counters.{{fn_name}}, start_stop_timer(&timer));

{{endfnall}}


/* MPI_Pcontrol needs special handling, as it doesn't call a PMPI function and
 * PnMPI does not implement the required XMPI call. Instead PnMPI will act as a
 * multiplexer for MPI_Pcontrol, so all we have to do is increment the counter
 * and return. */

int MPI_Pcontrol(const int level, ...)
{
  metric_atomic_inc(counters.MPI_Pcontrol);

  return MPI_SUCCESS;
}


/** \brief Print the statistics.
 *
 * \details This function will print the statistics to stdout for each rank and
 *  a sum of all ranks.
 *
 * \todo In feature releases this should be done by the PNMPI_AppShutdown hook
 *  to track duplicated MPI_Finalize calls.
 *
 *
 * \return The return value of PMPI_Finalize will be pass through.
 */
int MPI_Finalize()
{
  /* Flush the buffers to avoid fragments in the output.
   *
   * NOTE: This can't fully prevent buffered output being displayed between
   *       fresh one, as there's no guaranty the MPI sends the buffers back
   *       to the host immediately. */
  fflush(stdout);
  fflush(stderr);


  int rank, size;

  PMPI_Comm_rank_assert(MPI_COMM_WORLD, &rank);
  PMPI_Comm_size_assert(MPI_COMM_WORLD, &size);

  /* The entire output will be handled by rank 0 to avoid output collisions, as
   * the individual ranks can't sync their output being displayed at the
   * frontend. Therefore, rank 0 prints a header and its counters first. */
  if (rank == 0)
    {
      printf("\n\n################################\n\n"
             "MPI call stats:\n\n"
             " Rank 0:\n");
      print_counters(&counters);
    }

  /* Rank 0 should receive the counters from other ranks now, to display those.
   * These will be summed up for the total counters printed below, to cache the
   * counters instead of using a reduction for receiving the counters of all
   * ranks a second time. */
  struct counter tmp = { 0 };
  int n;
  for (n = 1; n < size; n++)
    {
      {{forallfn fn_name MPI_Finalize}}
        if (rank > 0)
          PMPI_Send(&(counters.{{fn_name}}), 1, MPI_UNSIGNED_LONG, 0, 0,
                    MPI_COMM_WORLD);
        else
          {
            PMPI_Recv(&(tmp.{{fn_name}}), 1, MPI_UNSIGNED_LONG, n, 0,
                      MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            counters.{{fn_name}} += tmp.{{fn_name}};
          }
      {{endforallfn}}

      if (rank == 0)
        {
          printf("\n Rank %d:\n", n);
          print_counters(&tmp);
        }
    }

  /* Print the total counters. These have been summed up in the counter struct
   * of rank 0, so these have not to be received a second time. */
  if (rank == 0) {
    printf("\n Total:\n");
    print_counters(&counters);
    fflush(stdout);
  }


  /* Call original MPI_Finalize. */
  return XMPI_Finalize();
}
