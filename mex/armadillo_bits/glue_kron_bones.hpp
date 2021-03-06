// Copyright (C) 2009-2010 National ICT Australia (NICTA)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
// -------------------------------------------------------------------
// 
// Written by Conrad Sanderson - http://conradsanderson.id.au
// Written by Dimitrios Bouzas



//! \addtogroup glue_kron
//! @{



class glue_kron
  {
  public:

  template<typename eT> inline static void direct_kron(Mat<eT>&                out, const Mat<eT>&                A, const Mat<eT>&                B);
  template<typename T>  inline static void direct_kron(Mat< std::complex<T> >& out, const Mat< std::complex<T> >& A, const Mat<T>&                 B);
  template<typename T>  inline static void direct_kron(Mat< std::complex<T> >& out, const Mat<T>&                 A, const Mat< std::complex<T> >& B);
  
  template<typename T1, typename T2>   inline static void apply(Mat<typename T1::elem_type>& out, const Glue<T1,T2,glue_kron>& X);
  };



//! @}

