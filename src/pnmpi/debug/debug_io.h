/* This file is part of P^nMPI.
 *
 * Copyright (c)
 *  2008-2016 Lawrence Livermore National Laboratories, United States of America
 *  2011-2016 ZIH, Technische Universitaet Dresden, Federal Republic of Germany
 *  2013-2016 RWTH Aachen University, Federal Republic of Germany
 *
 *
 * P^nMPI is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation version 2.1 dated February 1999.
 *
 * P^nMPI is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with P^nMPI; if not, write to the
 *
 *   Free Software Foundation, Inc.
 *   51 Franklin St, Fifth Floor
 *   Boston, MA 02110, USA
 *
 *
 * Written by Martin Schulz, schulzm@llnl.gov.
 *
 * LLNL-CODE-402774
 */

#ifndef PNMPI_PRINT_H
#define PNMPI_PRINT_H


#include <stdlib.h>

#include <pnmpi/attributes.h>


/** \brief Debug levels for \ref pnmpi_debug.
 */
enum pnmpi_debug_level
{
  /** \brief Messages about PnMPI initialization (before MPI is initialized).
   *
   * \note Users should not enable this level when running PnMPI with more than
   *  one rank, as messages will be printed on all ranks, because PnMPI can't
   *  get the rank of the process at this time yet.
   */
  PNMPI_DEBUG_INIT = (1 << 0),

  /** \brief Messages about module loading.
   *
   * \details This log level is used for information about the loaded modules,
   *  what functions they provide and their settings.
   *
   * \note Users should not enable this level when running PnMPI with more than
   *  one rank, as messages will be printed on all ranks, because PnMPI can't
   *  get the rank of the process at this time yet.
   */
  PNMPI_DEBUG_MODULE = (1 << 1),

  /** \brief Messages about MPI call entry and exit.
   *
   * \details This log level is used for information about entry and exit points
   *  of MPI calls of the different layers.
   */
  PNMPI_DEBUG_CALL = (1 << 2)
};


/** \brief Macro to use for debug printing.
 *
 * \details To optimize the code for speed, debug printing can be disabled with
 *  the PNMPI_NO_DEBUG compile flag.
 */
#ifndef PNMPI_NO_DEBUG
#define pnmpi_debug(...) pnmpi_print_debug(__VA_ARGS__)
#else
#define pnmpi_debug(...)
#endif


/** \brief Print an error message with file and line number and exit.
 *
 * \details This macro is used as shortcut to print a warning with the current
 *  function and line number and exiting the program with an error code after
 *  printing the message.
 *
 *
 * \param format Printf-like format string.
 * \param ... Arguments to be evaluated.
 */
#define pnmpi_error(format, ...)                                            \
  {                                                                         \
    pnmpi_warning("%s:%d: " format, __FUNCTION__, __LINE__, ##__VA_ARGS__); \
    exit(EXIT_FAILURE);                                                     \
  }


/* The PnMPI API should be C++ compatible, too. We have to add the extern "C"
 * stanza to avoid name mangeling. Otherwise C++ PnMPI modules would not find
 * PnMPI API functions. */
#ifdef __cplusplus
extern "C" {
#endif


PNMPI_FUNCTION_PRINTF(2, 3)
PNMPI_FUNCTION_ARG_NONNULL(2)
void pnmpi_print_debug(const enum pnmpi_debug_level level, const char *format,
                       ...);

PNMPI_FUNCTION_PRINTF(1, 2)
PNMPI_FUNCTION_ARG_NONNULL(1)
void pnmpi_warning(const char *format, ...);
///< \hidecallergraph


#ifdef __cplusplus
}
#endif


#endif
