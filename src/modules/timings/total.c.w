
#include <stdio.h>

#include <stddef.h>
#include <sys/time.h>

#include <mpi.h>
#include <pnmpi/debug_io.h>
#include <pnmpi/hooks.h>
#include <pnmpi/private/pmpi_assert.h>
#include <pnmpi/private/tls.h>
#include <pnmpi/service.h>
#include <pnmpi/xmpi.h>

#include "pnmpi-atomic.h"

typedef unsigned long long timing_t;

timing_t get_time_ns() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000000000ll + tv.tv_usec * 1000ll;
}



/* If there is no atomic support or no thread local storage support, we'll limit
 * the threading level of this module to MPI_THREAD_SERIALIZED, so it is safe to
 * use with threaded applications (but they may become slower!). */
#if defined(METRIC_NO_ATOMIC) || defined(PNMPI_COMPILER_NO_TLS)
const int PNMPI_SupportedThreadingLevel = MPI_THREAD_SERIALIZED;
#endif


/** \brief Struct of timing storage.
 *
 * \details This struct stores the time spent for each MPI calls.
 */
static struct timing_storage
{
  {{forallfn fn_name}}
  metric_atomic_keyword timing_t {{fn_name}};
  {{endforallfn}}
} timing_storage = { 0 };

/** \brief Helper function to print \ref timing_storage struct.
 *
 * \details This function will print all functions called at least one
 *  nanosecond.
 *
 *
 * \param t \ref timing_storage struct to be printed.
 */
static void print_counters(struct timing_storage *t)
{
  {{forallfn fn_name}}
  if (t->{{fn_name}} > 0)
    printf("  %13.9fs %s\n", t->{{fn_name}} * 0.000000001, "{{fn_name}}");
  {{endforallfn}}
}

static timing_t get_sum_of_counters(struct timing_storage *t)
{
  timing_t sum = 0;
  {{forallfn fn_name}}
  if (t->{{fn_name}} > 0)
    sum += t->{{fn_name}};
  {{endforallfn}}

  return sum;
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

/** \brief Global variable to store total runtime.
 */
timing_t total_runtime = 0;


/** \brief PnMPI module initialization hook.
 *
 * \details This function sets all counters to zero and initializes the module.
 */
void PNMPI_Init()
{
  total_runtime = get_time_ns();

  /* The timing statics should be printed after all MPI_Finalize calls of the
   * following PnMPI levels and stacks have been finished. As the timers may be
   * invoked at different levels to track only some of the PnMPI levels, we have
   * to count how often this module is in the PnMPI stack to know how many
   * MPI_Finalize calls we have to ignore. */
  //metric_atomic_inc(metric_invocations);


  /* Timing for MPI_Pcontrol is available for metric-timing invocations before
   * and after the modules to test only. To satisfy this, we'll check for each
   * invocation of this module, if it's Pcontrol-enabled. If it is, a counter
   * will be increased. It will be checked when calling MPI_Pcontrol. */

   PNMPI_Service_GetPcontrolSelf();
  // if (PNMPI_Service_GetPcontrolSelf())
  //   metric_atomic_inc(metric_invocations_pcontrol);
}


{{fnall fn_name MPI_Finalize MPI_Pcontrol}}
  static pnmpi_compiler_tls_keyword timing_t timer = 0;

  metric_atomic_add(timing_storage.{{fn_name}}, start_stop_timer(&timer));
  WRAP_MPI_CALL_PREFIX
  int ret = X{{fn_name}}({{args}});
  WRAP_MPI_CALL_POSTFIX
  metric_atomic_add(timing_storage.{{fn_name}}, start_stop_timer(&timer));

  return ret;
{{endfnall}}


/* MPI_Pcontrol needs special handling, as it doesn't call a PMPI function and
 * PnMPI does not implement the required XMPI call. Instead PnMPI will act as a
 * multiplexer for MPI_Pcontrol, so all we have to do is increment the counter
 * and return.
 *
 * The second problem is, we can't measure the timings in simple mode, as we
 * need a start- and end-point to measure the time, but PnMPI gives us (per
 * default) only the start-point. To overcome this metric_invocations_pcontrol
 * must be devideable by two, so the first call is our start- and the second the
 * end-time.
 *
 * Note: This option requires both modules set pcontrol to 'on'! */

int MPI_Pcontrol(const int level, ...)
{
  /* At this point it is save to get metric_invocations without any atomic
   * safety, as writes only occur in PnMPI_Registration_Point. */

  // if ((metric_invocations_pcontrol % 2) != 0)
  //   PNMPI_Error("metric-timing can measure the time of MPI_Pcontrol in "
  //               "advanced mode, only.\n");


  /* The local timer will be defined as thread local storage, as each thread may
   * call the function to be measured at the same time. With thread local
   * storage, each thread has its own timer and does not get into conflict with
   * other threads.
   *
   * This variable also will be used to indicate, if the timer is active. If the
   * timer is 0, it is inactive, otherise it is active. */
  static pnmpi_compiler_tls_keyword timing_t timer = 0;
  metric_atomic_add(timing_storage.MPI_Pcontrol, start_stop_timer(&timer));

  return MPI_SUCCESS;
}


/** \brief Print the statistics.
 *
 * \details This function will print the statistics to stdout for each rank and
 *  a sum of all ranks.
 *
 *
 * \return The return value of PMPI_Finalize will be pass through.
 */
int MPI_Finalize()
{
  // static pnmpi_compiler_tls_keyword timing_t timer = 0;
  // metric_atomic_add(timing_storage.MPI_Finalize, start_stop_timer(&timer));
  // int ret = XMPI_Finalize();
  // metric_atomic_add(timing_storage.MPI_Finalize, start_stop_timer(&timer));

  fflush(stdout);
  fflush(stderr);


  int rank, size;

  PMPI_Comm_rank_assert(MPI_COMM_WORLD, &rank);
  PMPI_Comm_size_assert(MPI_COMM_WORLD, &size);

  total_runtime = get_time_ns() - total_runtime;
  timing_t mpi_t = get_sum_of_counters(&timing_storage);
  timing_t comp_t = total_runtime - mpi_t;
  timing_t max_comp_t, sum_comp_t;

  double commE = (double)comp_t/(double)total_runtime;
  double result_commE;
  PMPI_Reduce(&commE, &result_commE, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  double result_LB, result_PE;
  PMPI_Reduce(&comp_t, &max_comp_t, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
  PMPI_Reduce(&comp_t, &sum_comp_t, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  result_LB = ((double)sum_comp_t/size)/(double)max_comp_t;

  result_PE = result_commE * result_LB;





//  if (rank == 0) {
//      printf("\n Rank %d:\n", 0);
//      print_counters(&timing_storage);
//    }
//
//  for (int n = 1; n < size; n++) {
//    struct timing_storage tmp = { 0 };
//    {{forallfn fn_name MPI_Finalize}}
//      if (rank != 0) {
//        PMPI_Send(&(timing_storage.{{fn_name}}), 1, MPI_UNSIGNED_LONG_LONG, 0, 0,
//                  MPI_COMM_WORLD);
//      } else {
//          PMPI_Recv(&(tmp.{{fn_name}}), 1, MPI_UNSIGNED_LONG_LONG, n, 0,
//                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//      }
//    {{endforallfn}}
//  
//    if (rank == 0) {
//      printf("\n Rank %d:\n", n);
//      print_counters(&tmp);
//    }
//  }






//  if (rank == 0) {
//    printf("\n Rank %d:\n", 0);
//    printf("  %13.9f %s\n", 100 * commE, "Local Communication Efficiency");
//    printf("  %13.9f %s\n", comp_t * 0.000000001, "Local Computation time sec");
//    printf("  %13.9f %s\n", mpi_t * 0.000000001, "Local MPI time sec");
//    printf("  %13.9f %s\n", total_runtime * 0.000000001, "Local total runtime sec");
//  }
//
//  for (int n = 1; n < size; n++){
//    if (rank != 0) {
//      PMPI_Send(&commE, 1, MPI_DOUBLE, 0, 1,
//                  MPI_COMM_WORLD);
//      PMPI_Send(&comp_t, 1, MPI_UNSIGNED_LONG_LONG, 0, 2,
//                  MPI_COMM_WORLD);
//      PMPI_Send(&mpi_t, 1, MPI_UNSIGNED_LONG_LONG, 0, 3,
//                  MPI_COMM_WORLD);
//      PMPI_Send(&total_runtime, 1, MPI_UNSIGNED_LONG_LONG, 0, 4,
//                  MPI_COMM_WORLD);
//    } else {
//      printf("\n Rank %d:\n", n);
//
//      double metric_recv_buf;
//      timing_t time_recv_buf;
//
//      PMPI_Recv(&metric_recv_buf, 1, MPI_DOUBLE, n, 1,
//                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//      printf("  %13.9f %s\n", 100 * metric_recv_buf, "Local Communication Efficiency");
//
//      PMPI_Recv(&time_recv_buf, 1, MPI_UNSIGNED_LONG_LONG, n, 2,
//                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//      printf("  %13.9f %s\n", time_recv_buf * 0.000000001, "Local Computation time sec");
//
//      PMPI_Recv(&time_recv_buf, 1, MPI_UNSIGNED_LONG_LONG, n, 3,
//                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//      printf("  %13.9f %s\n", time_recv_buf * 0.000000001, "Local MPI time sec");
//
//      PMPI_Recv(&time_recv_buf, 1, MPI_UNSIGNED_LONG_LONG, n, 4,
//                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//      printf("  %13.9f %s\n", time_recv_buf * 0.000000001, "Local total runtime sec");
//    }
//  }




  if (rank == 0) {
    printf("\n%.2f\% %s\n", 100 * result_PE, "Parallel Efficiency");
    printf("    %.2f\% %s\n", 100 * result_commE, "Communication Efficiency");
    printf("    %.2f\% %s\n", 100 * result_LB, "Load Balance Efficiency");
    fflush(stdout);
  }



  int ret = XMPI_Finalize();

  return ret;
}
