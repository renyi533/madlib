/* ----------------------------------------------------------------------- *//**
 *
 * @file rt.cpp
 *
 * @brief regression tree related functions
 *
 *//* ----------------------------------------------------------------------- */

#include <dbconnector/dbconnector.hpp>
#include <modules/prob/boost.hpp>
#include <modules/prob/student.hpp>
#include <modules/shared/HandleTraits.hpp>

#include "rt.hpp"

namespace madlib {

namespace modules {

namespace cart {

/*
 * @brief This function returns the values of selected features.
 * 
 * @param fvals  The vector containing the values of all features.
 * @param fids   The vector containing the ids of selected features.
 * 
 * @return The vector containing the values of selected features.
 */
AnyType get_selected_feature::run(AnyType & args)
{
    ArrayHandle<float8> fval_array  =   args[0].getAs< ArrayHandle<float8> >();
    ArrayHandle<int32>  fid_array   =   args[1].getAs< ArrayHandle<int32> >();
    int32_t             id_count    =   fid_array.size();
    int32_t             val_count   =   fval_array.size();
	ArrayBuildState     build_state;
	Oid                 elem_typ = FLOAT8OID;
	int32_t             iterator_idx;

	get_typlenbyvalalign
        (
         elem_typ, 
         &build_state.typlen, 
         &build_state.typbyval, 
         &build_state.typalign
        );

    build_state.mcontext = NULL;
    build_state.alen    = id_count;
    build_state.dvalues = (Datum *) palloc(build_state.alen * sizeof(Datum));
    build_state.dnulls  = NULL;
    build_state.nelems  = build_state.alen;
    build_state.element_type = elem_typ;

    MutableArrayHandle<float8>  outarray(
        construct_array(build_state.dvalues,build_state.nelems,
                        build_state.element_type,build_state.typlen,
                        build_state.typbyval,build_state.typalign)
    );

    for (iterator_idx = 0; iterator_idx < id_count; iterator_idx++)
    {
        if ((int32)fid_array[iterator_idx] == 0)
            outarray[iterator_idx]   = 1;
        else
        {
            if (fid_array[iterator_idx] < 1 ||
                    fid_array[iterator_idx] > val_count)
                throw std::invalid_argument("invalid feature id");

            outarray[iterator_idx] = fval_array[fid_array[iterator_idx]-1];
        }
    }
    return  outarray;
}

} // namespace cart

} // namespace modules

} // namespace madlib
