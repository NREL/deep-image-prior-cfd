#!/bin/bash

function runcase () {
    pelec="${1}"
    rundir="${2}"
    probin_fname="${3}"
    input_fname="${4}"
    bin_fname="${5}"

    echo "================================================================================"
    echo "Running case ${rundir}"
    mkdir -p "${rundir}"
    cp "${probin_fname}" "${rundir}"
    cp "${input_fname}" "${rundir}"
    cp "${bin_fname}" "${rundir}/input.in"

    fromdir=$(pwd)
    cd "${rundir}" || exit
    rm -rf chk* plt* datlog ic.txt
    mpirun -np 1 ${pelec} inputs_2d
    cd "${fromdir}" || exit
}

# Files and paths
workdir=`pwd`
pelecbin="${workdir}/PeleC2d.gnu.MPI.ex"
tdir="${workdir}/turbulence"
resultdir="${tdir}/results"
probin="${tdir}/probin"
input="${tdir}/inputs_2d"

# Cases to runs
casedirs=("case000008" "case000009" "case000010" "case000011" "case000012")

for casedir in "${casedirs[@]}"
do
    fdir="${resultdir}/${casedir}"
    echo "Running Pele for ${fdir}"

    # run original case
    odir="${fdir}/original"
    bin_name="${fdir}/original.in"
    runcase "${pelecbin}" "${odir}" "${probin}" "${input}" "${bin_name}"

    # run reconstructed case
    odir="${fdir}/result"
    bin_name="${fdir}/result.in"
    runcase "${pelecbin}" "${odir}" "${probin}" "${input}" "${bin_name}"

    # run interpolated case
    odir="${fdir}/interp"
    bin_name="${fdir}/interp.in"
    runcase "${pelecbin}" "${odir}" "${probin}" "${input}" "${bin_name}"
done
