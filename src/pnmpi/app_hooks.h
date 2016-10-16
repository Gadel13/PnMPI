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

#ifndef PNMPI_APP_HOOKS_H
#define PNMPI_APP_HOOKS_H


/** \brief Type of MPI interface to be used by the application.
 */
typedef enum pnmpi_mpi_interface {
  PNMPI_INTERFACE_C,      ///< Application uses the C MPI interface.
  PNMPI_INTERFACE_Fortran ///< Application uses the Fortran MPI interface.
} pnmpi_mpi_interface;


/** \brief Cache for 'provided' return value of PMPI_Init_thread.
 *
 * \details PMPI_Init_thread returns the provided MPI threading level. The value
 *  will be stored in this variable, to pass it to the application in later
 *  calls to \ref MPI_Init_thread.
 */
extern int pnmpi_mpi_thread_level_provided;


pnmpi_mpi_interface pnmpi_get_mpi_interface();


#endif
