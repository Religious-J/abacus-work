#include "diago_iter_assist.h"

#include "module_base/blas_connector.h"
#include "module_base/complexmatrix.h"
#include "module_base/constants.h"
#include "module_base/global_variable.h"
#include "module_base/lapack_connector.h"
#include "module_base/module_device/device.h"
#include "module_base/parallel_reduce.h"
#include "module_base/timer.h"
#include "module_hsolver/kernels/dngvd_op.h"
#include "module_hsolver/kernels/math_kernel_op.h"

#include <ostream>

namespace hsolver
{

// 子空间中的哈密顿对角化
//----------------------------------------------------------------------
// Hamiltonian diagonalization in the subspace spanned
// by nstart states psi (atomic or random wavefunctions).
// Produces on output n_band eigenvectors (n_band <= nstart) in evc.
//----------------------------------------------------------------------
template <typename T, typename Device>
void DiagoIterAssist<T, Device>::diagH_subspace(hamilt::Hamilt<T, Device>* pHamilt, // hamiltonian operator carrier
                                                const psi::Psi<T, Device>& psi,     // [in] wavefunction
                                                psi::Psi<T, Device>& evc,           // [out] wavefunction
                                                Real* en,                           // [out] eigenvalues
                                                int n_band // [in] number of bands to be calculated, also number of rows
                                                           // of evc, if set to 0, n_band = nstart, default 0
)
// 波函数（`psi`）和输出特征向量（`evc`）由`Psi`类表示。哈密顿量（`pHamilt`）被抽象为提供必要操作的运算符载体。
{
    ModuleBase::TITLE("DiagoIterAssist", "diagH_subspace");
    ModuleBase::timer::tick("DiagoIterAssist", "diagH_subspace");

    // two case:
    // 1. pw base: nstart = n_band, psi(nbands * npwx)
    // 2. lcao_in_pw base: nstart >= n_band, psi(NLOCAL * npwx)
    const int nstart = psi.get_nbands();  // nbands
    if (n_band == 0)
        n_band = nstart;
    assert(n_band <= nstart);

    // 该函数使用自定义的内存处理操作`resmem_complex_op`动态分配相关子空间的哈密顿矩阵、重叠矩阵和系数矩阵的方阵。
    T *hcc = nullptr, *scc = nullptr, *vcc = nullptr;
    resmem_complex_op()(ctx, hcc, nstart * nstart, "DiagSub::hcc");
    resmem_complex_op()(ctx, scc, nstart * nstart, "DiagSub::scc");
    resmem_complex_op()(ctx, vcc, nstart * nstart, "DiagSub::vcc");
    setmem_complex_op()(ctx, hcc, 0, nstart * nstart);
    setmem_complex_op()(ctx, scc, 0, nstart * nstart);
    setmem_complex_op()(ctx, vcc, 0, nstart * nstart);

    const int dmin = psi.get_current_nbas();  // current_k 的当前 basis 数
    const int dmax = psi.get_nbasis();        // basis 数

    // qianrui improve this part 2021-3-14
    const T* ppsi = psi.get_pointer();       // ppsi -> psi

    std::cout << "############################" << std::endl;

    // 对`psi`执行完整的哈密顿量和重叠操作，得到临时乘积`hphi`和`sphi`，这些乘积用于构建子空间哈密顿矩阵（`hcc`）和重叠矩阵（`scc`）

    // allocated hpsi
    // std::vector<T> hpsi(psi.get_nbands() * psi.get_nbasis());

    // T* hphi = nullptr;
    // resmem_complex_op()(ctx, hphi, psi.get_nbasis(), "DiagSub::hpsi");
    // setmem_complex_op()(ctx, hphi, 0,  psi.get_nbasis());

    T* hphi = nullptr;
    // resmem_complex_op()(ctx, hphi, psi.get_nbands() * psi.get_nbasis(), "DiagSub::hpsi");
    // setmem_complex_op()(ctx, hphi, 0, psi.get_nbands() * psi.get_nbasis());
    resmem_complex_op()(ctx, hphi, psi.get_nbasis(), "DiagSub::hpsi");
    setmem_complex_op()(ctx, hphi, 0, psi.get_nbasis());

    // do hPsi for all bands
    // psi::Range all_bands_range(1, psi.get_current_k(), 0, psi.get_nbands() - 1);
    // hpsi_info hpsi_in(&psi, all_bands_range, hphi);
    // pHamilt->ops->hPsi(hpsi_in);

    // do hPsi for all band by band
    for (int i = 0; i < psi.get_nbands(); i++){
       // Psi(nks, nbands, nbasis)
    //    setmem_complex_op()(ctx, hphi, 0, psi.get_nbands() * psi.get_nbasis());
       psi::Range band_by_band_range(1, psi.get_current_k(), i, i);
       hpsi_info hpsi_in(&psi, band_by_band_range, hphi);
       pHamilt->ops->hPsi(hpsi_in);
       
    //    T* cur = hphi + i * psi.get_nbasis();
       T* cur = hphi;
       T* cur2 = hcc + i * nstart;

       gemv_op<T, Device>()(
            ctx,
            'C',
            dmax,  
            nstart,  
            &one,
            ppsi,
            dmax,  // nbasis
            cur,
            1,
            &zero,
            cur2,
            1
       );
    }

    // --- print --- //
    // std::cout << "***" << "nstart = " << nstart
    //         << " dmin = " << dmin 
    //         << " dmax = " << dmax;

    // std::cout << std::endl;
     
    // nstart 8
    // dmin =  2685
    // dmax =  2730
    // ------------- //

    // gemm_op<T, Device>()(
    //     ctx,
    //     'C',
    //     'N',
    //     nstart, // 8
    //     nstart, // 8
    //     dmin,   // 2685
    //     &one,
    //     ppsi,  // 8 * 2730 
    //     dmax,  // 2730
    //     hphi,  // 8 * 2730
    //     dmax,  // 2730
    //     &zero,
    //     hcc,  // 8 * 8
    //     nstart
    // );

// print
    // std::cout << "hcc: " << std::endl;
    // int num=1;
    // for(int i=0; i < nstart*nstart; i++){
    //      std::cout << *(hcc+i) << " "; 
    //      if(num++ % 8 == 0) std::cout << std::endl;
    // }
    // std::cout << std::endl;

    // for(int i=0; i < psi.get_nbands() * psi.get_nbasis(); i++){
    //     std::cout << "hphi: " << *(hphi+i) << " "; 
    // }
    // std::cout << std::endl;

    // for(int i=0; i < psi.get_nbands() * psi.get_nbasis(); i++){
    //     std::cout << "ppsi: " << *(ppsi+i) << " "; 
    // }
    // std::cout << std::endl;

    delmem_complex_op()(ctx, hphi);

    // allocated spsi
    T* sphi = nullptr;
    resmem_complex_op()(ctx, sphi, nstart * dmax, "DiagSub::spsi");
    setmem_complex_op()(ctx, sphi, 0, nstart * dmax);
    // do sPsi for all bands
    pHamilt->sPsi(ppsi, sphi, dmax, dmin, nstart);

    gemm_op<T, Device>()(ctx, 'C', 'N', nstart, nstart, dmin, &one, ppsi, dmax, sphi, dmax, &zero, scc, nstart);
    delmem_complex_op()(ctx, sphi);

    //  如果代码在并行环境中运行（基于MPI），则通过检查`NPROC_IN_POOL`全局变量，对结果矩阵进行全局归约。
    if (GlobalV::NPROC_IN_POOL > 1)
    {
        Parallel_Reduce::reduce_pool(hcc, nstart * nstart);
        Parallel_Reduce::reduce_pool(scc, nstart * nstart);
    }

    // 调用LAPACK来解决广义特征问题`Hv=SvE`，其中`H`是子空间哈密顿矩阵，`S`是子空间重叠矩阵，`E`是特征值的对角矩阵，`v`是对应于子空间解的特征向量。
    // after generation of H and S matrix, diag them
    DiagoIterAssist::diagH_LAPACK(nstart, n_band, hcc, scc, nstart, en, vcc);

    //=======================
    // diagonize the H-matrix
    //=======================

    // 矩阵`evc`现在收集对应于描述占据电子态所需的最小特征值的特征向量，这取决于由BASIS_TYPE和CALCULATION全局变量确定的条件。
    if (((GlobalV::BASIS_TYPE == "lcao") || (GlobalV::BASIS_TYPE == "lcao_in_pw")) && (GlobalV::CALCULATION == "nscf"))
    {
        GlobalV::ofs_running << " Not do zgemm to get evc." << std::endl;
    }
    else if (((GlobalV::BASIS_TYPE == "lcao") || (GlobalV::BASIS_TYPE == "lcao_in_pw"))
             && ((GlobalV::CALCULATION == "scf") || (GlobalV::CALCULATION == "md")
                 || (GlobalV::CALCULATION == "relax"))) // pengfei 2014-10-13
    {
        // because psi and evc are different here,
        // I think if psi and evc are the same,
        // there may be problems, mohan 2011-01-01
        gemm_op<T, Device>()(ctx,
                             'N',
                             'N',
                             dmax,
                             n_band,
                             nstart,
                             &one,
                             ppsi, // dmax * nstart
                             dmax,
                             vcc, // nstart * n_band
                             nstart,
                             &zero,
                             evc.get_pointer(),
                             dmax);
    }
    else
    {
        // As the evc and psi may refer to the same matrix, we first
        // create a temporary matrix to store the result. (by wangjp)
        // qianrui improve this part 2021-3-13
        T* evctemp = nullptr;
        resmem_complex_op()(ctx, evctemp, n_band * dmin, "DiagSub::evctemp");
        setmem_complex_op()(ctx, evctemp, 0, n_band * dmin);

        gemm_op<T, Device>()(ctx,
                             'N',
                             'N',
                             dmin,
                             n_band,
                             nstart,
                             &one,
                             ppsi, // dmin * nstart
                             dmax,
                             vcc, // nstart * n_band
                             nstart,
                             &zero,
                             evctemp,
                             dmin);

        matrixSetToAnother<T, Device>()(ctx, n_band, evctemp, dmin, evc.get_pointer(), dmax);
        // for (int ib = 0; ib < n_band; ib++)
        // {
        //     for (int ig = 0; ig < dmin; ig++)
        //     {
        // evc(ib, ig) = evctmp(ib, ig);
        //     }
        // }
        delmem_complex_op()(ctx, evctemp);
    }

    delmem_complex_op()(ctx, hcc);
    delmem_complex_op()(ctx, scc);
    delmem_complex_op()(ctx, vcc);

    ModuleBase::timer::tick("DiagoIterAssist", "diagH_subspace");
}

// 初始化基于给定子空间的哈密顿矩阵对角化
// 初始化和对角化一个子空间哈密顿量，计算重叠乘积，并更新量子力学模拟中的特征向量和特征值
template <typename T, typename Device>
void DiagoIterAssist<T, Device>::diagH_subspace_init(
    hamilt::Hamilt<T, Device>* pHamilt, // 一个指向哈密顿矩阵对象的指针，用于执行对角化操作
    const T* psi,                       // 一个指向波函数的指针，表示哈密顿量作用的对象。
    int psi_nr,                         // 表示波函数矩阵的行数   nbands
    int psi_nc,                         // 列数x                nbasis
    psi::Psi<T, Device>& evc,           // 量子力学模拟中的特征向量和特征值对象
    Real* en                            // 实数类型的数组，用于存储特征值
)
{

    std::cout << "YYYYYYYYYYYYYYYYYYYYYYYYYYYYYES!!!!!!!!" << std::endl;

    ModuleBase::TITLE("DiagoIterAssist", "diagH_subspace_init");
    ModuleBase::timer::tick("DiagoIterAssist", "diagH_subspace_init");

    // two case:
    // 1. pw base: nstart = n_band, psi(nbands * npwx)
    // 2. lcao_in_pw base: nstart >= n_band, psi(NLOCAL * npwx)

    const int nstart = psi_nr;           // 波函数矩阵的行数 nbands
    const int n_band = evc.get_nbands(); // 波函数的能带数

    Device* ctx = {};

    // 为不同的矩阵操作分配内存 有 hcc scc vcc
    T *hcc = nullptr, *scc = nullptr, *vcc = nullptr;
    resmem_complex_op()(ctx, hcc, nstart * nstart, "DiagSub::hcc");
    resmem_complex_op()(ctx, scc, nstart * nstart, "DiagSub::scc");
    resmem_complex_op()(ctx, vcc, nstart * nstart, "DiagSub::vcc");
    setmem_complex_op()(ctx, hcc, 0, nstart * nstart);
    setmem_complex_op()(ctx, scc, 0, nstart * nstart);
    setmem_complex_op()(ctx, vcc, 0, nstart * nstart);

    const int dmin = evc.get_current_nbas(); // current_k 的当前 basis 数
    const int dmax = evc.get_nbasis();       // basis 数

    // 创建一个临时的 psi_temp 并与输入的 psi 同步数据 - 相同的矩阵类型  
    psi::Psi<T, Device> psi_temp(1, nstart, psi_nc, &evc.get_ngk(0));   // psi_nr * psi_nc  |  ngk -> ?
    syncmem_complex_op()(ctx, ctx, psi_temp.get_pointer(), psi, psi_temp.size());

    // 指针指向 psi_temp
    const T* ppsi = psi_temp.get_pointer();  // mutable

    // 分配一个数组 hpsi 来存储 哈密顿量 * psi 的乘积
    // 调用 hPsi 来计算哈密顿量作用于波函数的结果
    // **
    // T* hpsi = nullptr; 
    // resmem_complex_op()(ctx, hpsi, psi_temp.get_nbands() * psi_temp.get_nbasis(), "DiagSub::hpsi");
    // setmem_complex_op()(ctx, hpsi, 0, psi_temp.get_nbands() * psi_temp.get_nbasis());

    T* hpsi = nullptr;
    resmem_complex_op()(ctx, hpsi, psi_temp.get_nbasis(), "DiagSub::hpsi");
    setmem_complex_op()(ctx, hpsi, 0,  psi_temp.get_nbasis());

    // ================================================
    // std::vector<T> hpsi(psi_temp.get_nbands() * psi_temp.get_nbasis());

    // 调用提供的 `Hamilt<T, Device>` 操作符（`pHamilt`）的 `hPsi` 函数，使哈密顿量作用于 psi，结果存储在 `hpsi` 中。

    // 根据是在 GPU 还是 CPU 上运行
    // psi 数据要么按带逐带处理，要么一次性处理所有带

// ------------ before -------------- //
//     // do hPsi for all bands
//     if (base_device::get_device_type(ctx) == base_device::GpuDevice) // GPU
//     {
//         // 循环遍历所有能带 bands
//         for (int i = 0; i < psi_temp.get_nbands(); i++)
//         {
//             // 创建了一个波段范围对象，用于表示当前循环中的能带
// /*
// Range::Range(const bool k_first_in, const size_t index_1_in, const size_t range_1_in, const size_t range_2_in)
// {
//     k_first = k_first_in;
//     index_1 = index_1_in;
//     range_1 = range_1_in;
//     range_2 = range_2_in;
// }
// */          // Psi(nks, nbands, nbasis)
//             psi::Range band_by_band_range(1, psi_temp.get_current_k(), i, i);
//             // 创建了一个用于存储哈密顿量作用结果的临时对象 hpsi_in，该对象包括了波函数对象、波段范围和存储结果的数组。
//             hpsi_info hpsi_in(&psi_temp, band_by_band_range, hpsi + i * psi_temp.get_nbasis());
//             // 算子
//             if (pHamilt->ops == nullptr)
//             {
//                 ModuleBase::WARNING("DiagoIterAssist::diagH_subspace_init",
//                                     "Severe warning: Operators in Hamilt are not allocated yet, will return value of "
//                                     "psi to evc directly\n");
//                 for (int iband = 0; iband < evc.get_nbands(); iband++)
//                 {
//                     for (int ig = 0; ig < evc.get_nbasis(); ig++)
//                     {
//                         evc(iband, ig) = psi[iband * evc.get_nbasis() + ig];
//                     }
//                     en[iband] = 0.0;
//                 }
//                 return;
//             }
//             // 调用哈密顿矩阵对象中的 hPsi 操作符，将哈密顿量作用于波函数的结果存储在 hpsi_in 中
//             pHamilt->ops->hPsi(hpsi_in);
//         }
//     }

    if (base_device::get_device_type(ctx) == base_device::GpuDevice) // GPU
    {
        for (int i = 0; i < psi_temp.get_nbands(); i++)
        {
          // Psi(nks, nbands, nbasis)
            psi::Range band_by_band_range(1, psi_temp.get_current_k(), i, i);
            hpsi_info hpsi_in(&psi_temp, band_by_band_range, hpsi);

            if (pHamilt->ops == nullptr)
            {
                ModuleBase::WARNING("DiagoIterAssist::diagH_subspace_init",
                                    "Severe warning: Operators in Hamilt are not allocated yet, will return value of "
                                    "psi to evc directly\n");
                for (int iband = 0; iband < evc.get_nbands(); iband++)
                {
                    for (int ig = 0; ig < evc.get_nbasis(); ig++)
                    {
                        evc(iband, ig) = psi[iband * evc.get_nbasis() + ig];
                    }
                    en[iband] = 0.0;
                }
                return;
            }
            // 调用哈密顿矩阵对象中的 hPsi 操作符，将哈密顿量作用于波函数的结果存储在 hpsi_in 中
            pHamilt->ops->hPsi(hpsi_in);

            gemv_op<T, Device>()(
                ctx,
                'C',
                dmax,  
                nstart,  
                &one,
                ppsi,
                dmax,  // nbasis
                hpsi,
                1,
                &zero,
                hcc + i*nstart,
                1
            );
        }
    }


// ------------- before ------------ // 
    // // CPU端
    // else if (base_device::get_device_type(ctx) == base_device::CpuDevice) // CPU
    // {
    //     // 创建波段范围对象: 使用 psi::Range 对象创建了一个所有波段的范围 all_bands_range。
    //     // 起始于1，结束于当前 k 点数量 psi_temp.get_current_k()
    //     // 这也设计到带号 nbands 的一个偏置范围从 0 到 psi_temp.get_nbands()-1。
    //     // all
    //     psi::Range all_bands_range(1, psi_temp.get_current_k(), 0, psi_temp.get_nbands() - 1);

    //     // 准备 hpsi_info 对象:
    //     // 创建了一个 hpsi_info 对象 hpsi_in，将 psi_temp 波函数对象和之前定义的 all_bands_range 波段范围传递成员初始化
    //     // 并和外部变量 hpsi 关联。
    //     hpsi_info hpsi_in(&psi_temp, all_bands_range, hpsi);

    //     // 检查算子:
    //     // 在调用之后必需的操作之前，检查了密度泛函理论的哈密顿量 pHamilt 里是否存储了必要的算子(ops)
    //     if (pHamilt->ops == nullptr)
    //     {
    //         // 执行算子
    //         ModuleBase::WARNING("DiagoIterAssist::diagH_subspace_init",
    //                             "Severe warning: Operators in Hamilt are not allocated yet, will return value of psi "
    //                             "to evc directly\n");
    //         for (int iband = 0; iband < evc.get_nbands(); iband++)
    //         {
    //             for (int ig = 0; ig < evc.get_nbasis(); ig++)
    //             {
    //                 evc(iband, ig) = psi[iband * evc.get_nbasis() + ig];
    //             }
    //             en[iband] = 0.0;
    //         }
    //         return;
    //     }
    //     // 调用哈密顿矩阵对象中的 hPsi 操作符，将哈密顿量作用于波函数的结果存储在 hpsi_in 中。
    //     pHamilt->ops->hPsi(hpsi_in);
    // }

    // CPU端
    else if (base_device::get_device_type(ctx) == base_device::CpuDevice) // CPU
    {

     std::cout << "CPU CPU CPU CPU CPU CPU" << std::endl;   
        for (int i = 0; i < psi_temp.get_nbands(); i++)
        {
            psi::Range band_by_band_range(1, psi_temp.get_current_k(), i, i);

            hpsi_info hpsi_in(&psi_temp, band_by_band_range, hpsi);

            if (pHamilt->ops == nullptr)
            {
                // 执行算子
                ModuleBase::WARNING("DiagoIterAssist::diagH_subspace_init",
                                "Severe warning: Operators in Hamilt are not allocated yet, will return value of psi "
                                "to evc directly\n");
                for (int iband = 0; iband < evc.get_nbands(); iband++)
                {
                    for (int ig = 0; ig < evc.get_nbasis(); ig++)
                    {
                        evc(iband, ig) = psi[iband * evc.get_nbasis() + ig];
                    }
                    en[iband] = 0.0;
                }
                return;
            }
            pHamilt->ops->hPsi(hpsi_in);

            gemv_op<T, Device>()(
                ctx,
                'C',
                dmax,  
                nstart,  
                &one,
                ppsi,
                dmax,  // nbasis
                hpsi,
                1,
                &zero,
                hcc + i*nstart,
                1
            );
        }
    }

    // print
    // std::cout << "***hcc: " << std::endl;
    // int num=1;
    // for(int i=0; i < nstart*nstart; i++){
    //      std::cout << *(hcc+i) << " "; 
    //      if(num++ % 8 == 0) std::cout << std::endl;
    // }
    // std::cout << std::endl;


    // 函数使用 BLAS 操作 gemm（通用矩阵乘法）两次执行涉及哈密顿量和重叠矩阵（位于 vcc 和 scc中）的矩阵乘法
    // 更新两个复数矩阵（hcc 和 scc）
    // c = alpha * op(a) * op(b) + beta * c
    // ppsi hpsi -> hcc
    // hcc=one⋅op(ppsi)⋅op(hpsi)+zero⋅hcc


    //     // ADD
    // gemv_op<T, Device>()(ctx,
    //                      'C',       // 传递共轭转置参数
    //                      dmax,      // A 的行数
    //                      dmin,      // A 的列数 dmin = nstart
    //                      &one,      // alpha
    //                      ppsi,      // A 矩阵
    //                      dmax,      
    //                      hpsi_,         // x 向量
    //                      1,         // x 的步进
    //                      &zero,     // beta
    //                      hcc_,         // y 向量
    //                      1);        // y 的步进


  // dmin x dmax * dmax x nstrat = dmin x nstrat = nstrat x nstart
    // gemm_op<T, Device>()(ctx,
    //                      'C',
    //                      'N',
    //                      nstart,
    //                      nstart,
    //                      nstart,  // dmin = evc.get_current_nbas();  current_k 的当前 basis 数
    //                      &one,
    //                      ppsi, // psi_temp H dmin * dmax  => nstart * dmax
    //                      dmax,
    //                      hpsi, // dmax * nstart       dmax = basis 数    nstart = psi_nr = bands 数
    //                      dmax,
    //                      &zero,
    //                      hcc, // nstart * nstart
    //                      nstart);

// // true
//     // dmin x dmax * dmax x nstrat
    // gemm_op<T, Device>()(ctx,
    //                      'C',
    //                      'N',
    //                      nstart,
    //                      nstart,
    //                      dmin,  // dmin = evc.get_current_nbas();  current_k 的当前 basis 数
    //                      &one,
    //                      ppsi, // psi_temp H dmin * dmax
    //                      dmax,
    //                      hpsi, // dmax * nstart       dmax = basis 数    nstart = psi_nr = bands 数
    //                      dmax,
    //                      &zero,
    //                      hcc, // nstart * nstart
    //                      nstart);

    // 把临时申请的 hpsi 的 free
    delmem_complex_op()(ctx, hpsi);


// --------------- spsi ---------------------- //

    // allocated spsi
    // 分配了用于存储哈密顿量 * psi 的乘积的临时数组 spsi
    // T* spsi = nullptr;
    // resmem_complex_op()(ctx, spsi, psi_temp.get_nbands() * psi_temp.get_nbasis(), "DiagSub::spsi");
    // setmem_complex_op()(ctx, spsi, 0, psi_temp.get_nbands() * psi_temp.get_nbasis());

    // // 其中 ppsi 指向 psi_temp
    // pHamilt->sPsi(ppsi, spsi, psi_temp.get_nbasis(), psi_temp.get_current_nbas(), psi_temp.get_nbands());

    // std::cout << "#################################3" << std::endl;

    // // 调用 gemm 通用矩阵乘法运算  ppsi spsi -> scc
    // gemm_op<T, Device>()(ctx, 'C', 'N', nstart, nstart, dmin, &one, ppsi, dmax, spsi, dmax, &zero, scc, nstart);

    // delmem_complex_op()(ctx, spsi);



    // T* spsi = hpsi;
    T* spsi = nullptr;
    resmem_complex_op()(ctx, spsi,  psi_temp.get_nbasis(), "DiagSub::spsi");
    setmem_complex_op()(ctx, spsi, 0,  psi_temp.get_nbasis());

    // 其中 ppsi 指向 psi_temp
    for(int i = 0; i < psi_temp.get_nbands(); i++){
        pHamilt->sPsi(ppsi+i*psi_temp.get_nbasis(), spsi, psi_temp.get_nbasis(), psi_temp.get_current_nbas(), 1);
        
        gemv_op<T, Device>()(
                ctx,
                'C',
                dmax,  
                nstart,  
                &one,
                ppsi,
                dmax,  // nbasis
                spsi,
                1,
                &zero,
                scc + i*nstart,
                1
            );
    }
    
    std::cout << "#################################3" << std::endl;

    // 调用 gemm 通用矩阵乘法运算  ppsi spsi -> scc
    // gemm_op<T, Device>()(ctx, 'C', 'N', nstart, nstart, dmin, &one, ppsi, dmax, spsi, dmax, &zero, scc, nstart);

    delmem_complex_op()(ctx, spsi);

    // delmem_complex_op()(ctx, hpsi);



    // print
    // std::cout << "***scc: " << std::endl;
    // int num=1;
    // for(int i=0; i < nstart*nstart; i++){
    //      std::cout << *(scc+i) << " "; 
    //      if(num++ % 8 == 0) std::cout << std::endl;
    // }
    // std::cout << std::endl;



    // 进行并行归约，可能是为了整合不同计算节点/池的结果
    if (GlobalV::NPROC_IN_POOL > 1)
    {
        Parallel_Reduce::reduce_pool(hcc, nstart * nstart);
        Parallel_Reduce::reduce_pool(scc, nstart * nstart);
    }

    // after generation of H and S matrix, diag them
    /// this part only for test, eigenvector would have different phase caused by micro numerical perturbation
    /// set 8 bit effective accuracy would help for debugging
    /*for(int i=0;i<nstart;i++)
    {
        for(int j=0;j<nstart;j++)
        {
            if(std::norm(hc(i,j))<1e-10) hc(i,j) = ModuleBase::ZERO;
            else hc(i,j) = std::complex<double>(double(int(hc(i,j).real()*100000000))/100000000, 0);
            if(std::norm(sc(i,j))<1e-10) sc(i,j) = ModuleBase::ZERO;
            else sc(i,j) = std::complex<double>(double(int(sc(i,j).real()*100000000))/100000000, 0);
        }
    }*/

    // 结果产生的 hcc（哈密顿量）矩阵和 scc（重叠）矩阵通过 diagH_LAPACK对角化，特征值存储在 en 中，特征向量存储在 `vcc` 中。
    DiagoIterAssist::diagH_LAPACK(nstart, n_band, hcc, scc, nstart, en, vcc);

    

    // print
    std::cout << "***vcc: " << std::endl;
    int num=1;
    for(int i=0; i < nstart*nstart; i++){
         std::cout << *(vcc+i) << " "; 
         if(num++ % 6 == 0) std::cout << std::endl;
    }
    std::cout << std::endl;


    //=======================
    // diagonize the H-matrix
    // 对 H 矩阵进行对角化
    //=======================

    if ((GlobalV::BASIS_TYPE == "lcao" || GlobalV::BASIS_TYPE == "lcao_in_pw") && GlobalV::CALCULATION == "nscf")
    {
        GlobalV::ofs_running << " Not do zgemm to get evc." << std::endl;
    }
    else if ((GlobalV::BASIS_TYPE == "lcao" || GlobalV::BASIS_TYPE == "lcao_in_pw"
              || (GlobalV::BASIS_TYPE == "pw" && GlobalV::psi_initializer))
             && (GlobalV::CALCULATION == "scf" || GlobalV::CALCULATION == "md"
                 || GlobalV::CALCULATION == "relax")) // pengfei 2014-10-13
    {
        // because psi and evc are different here,
        // I think if psi and evc are the same,
        // there may be problems, mohan 2011-01-01

        // 另一个 gemm 可能会进一步处理特征向量
        // gemm ppsi vcc -> evc
        gemm_op<T, Device>()(ctx,
                             'N',
                             'N',
                             dmax,   // 1
                             n_band, // 2
                             nstart,  // 3
                             &one,
                             ppsi, // dmax * nstart
                             dmax,  // 1
                             vcc, // nstart * n_band
                             nstart,  // 3
                             &zero,
                             evc.get_pointer(),
                             dmax);
    }
    else
    {
        // As the evc and psi may refer to the same matrix, we first
        // create a temporary matrix to store the result. (by wangjp)
        // qianrui improve this part 2021-3-13

        // 由于evc和psi可能引用同一个矩阵，我们首先创建一个临时矩阵来存储结果。

        T* evctemp = nullptr;
        resmem_complex_op()(ctx, evctemp, n_band * dmin, "DiagSub::evctemp");
        setmem_complex_op()(ctx, evctemp, 0, n_band * dmin);

        // gemm ppsi vcc -> evctemp
        gemm_op<T, Device>()(ctx,
                             'N',
                             'N',
                             dmin,
                             n_band,
                             nstart,
                             &one,
                             ppsi, // dmin * nstart
                             dmax,
                             vcc, // nstart * n_band
                             nstart,
                             &zero,
                             evctemp,
                             dmin);

        // 矩阵复制 evcttemp -> evc
        matrixSetToAnother<T, Device>()(ctx, n_band, evctemp, dmin, evc.get_pointer(), dmax);

        delmem_complex_op()(ctx, evctemp);
    }

    //  中间数组和矩阵 hcc、scc，vcc 被释放
    delmem_complex_op()(ctx, hcc);
    delmem_complex_op()(ctx, scc);
    delmem_complex_op()(ctx, vcc);
    ModuleBase::timer::tick("DiagoIterAssist", "diagH_subspace_init");
}

template <typename T, typename Device>
void DiagoIterAssist<T, Device>::diagH_LAPACK(const int nstart, // 起始矩阵维度
                                              const int nbands, // 需要对角化的波段数量
                                              const T* hcc,     // 哈密顿量矩阵的指针
                                              const T* scc,     // 重叠矩阵的指针
                                              const int ldh,    // nstart  // 阵的主维度（一般为 nstart）
                                              Real* e, // always in CPU // 特征值数组（始终在 CPU 上）
                                              T* vcc   // 特征向量矩阵的指针
)
{
    ModuleBase::TITLE("DiagoIterAssist", "diagH_LAPACK");
    ModuleBase::timer::tick("DiagoIterAssist", "diagH_LAPACK");

    // 为特征值分配内存并将其初始化为 0
    Real* eigenvalues = nullptr;
    resmem_var_op()(ctx, eigenvalues, nstart);
    setmem_var_op()(ctx, eigenvalues, 0, nstart);

    // 调用 dngvd_op 进行对角化操作
    dngvd_op<T, Device>()(ctx, nstart, ldh, hcc, scc, eigenvalues, vcc);

    // 如果设备类型是 GPU，则将特征值从 GPU 同步到 CPU
    if (base_device::get_device_type<Device>(ctx) == base_device::GpuDevice)
    {
#if ((defined __CUDA) || (defined __ROCM))
        // set eigenvalues in GPU to e in CPU
        syncmem_var_d2h_op()(cpu_ctx, gpu_ctx, e, eigenvalues, nbands);
#endif
    }

    // 如果设备类型是 CPU，则在 CPU 内同步特征值：
    else if (base_device::get_device_type<Device>(ctx) == base_device::CpuDevice)
    {
        // set eigenvalues in CPU to e in CPU
        syncmem_var_op()(ctx, ctx, e, eigenvalues, nbands);
    }

    delmem_var_op()(ctx, eigenvalues);

    // const bool all_eigenvalues = (nstart == nbands);
    // if (all_eigenvalues) {
    //     //===========================
    //     // calculate all eigenvalues
    //     //===========================
    //     // dngv_op<Real, Device>()(ctx, nstart, ldh, hcc, scc, res, vcc);
    //     dngvd_op<Real, Device>()(ctx, nstart, ldh, hcc, scc, res, vcc);
    // }
    // else {
    //     //=====================================
    //     // calculate only m lowest eigenvalues
    //     //=====================================
    //     dngvx_op<Real, Device>()(ctx, nstart, ldh, hcc, scc, nbands, res, vcc);
    // }

    ModuleBase::timer::tick("DiagoIterAssist", "diagH_LAPACK");
}

template <typename T, typename Device>
bool DiagoIterAssist<T, Device>::test_exit_cond(const int& ntry, const int& notconv)
{
    //================================================================
    // If this logical function is true, need to do diagH_subspace
    // and cg again.
    //================================================================

    bool scf = true;
    if (GlobalV::CALCULATION == "nscf")
        scf = false;

    // If ntry <=5, try to do it better, if ntry > 5, exit.
    const bool f1 = (ntry <= 5);

    // In non-self consistent calculation, do until totally converged.
    const bool f2 = ((!scf && (notconv > 0)));

    // if self consistent calculation, if not converged > 5,
    // using diagH_subspace and cg method again. ntry++
    const bool f3 = ((scf && (notconv > 5)));
    return (f1 && (f2 || f3));
}

template class DiagoIterAssist<std::complex<float>, base_device::DEVICE_CPU>;
template class DiagoIterAssist<std::complex<double>, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class DiagoIterAssist<std::complex<float>, base_device::DEVICE_GPU>;
template class DiagoIterAssist<std::complex<double>, base_device::DEVICE_GPU>;
#endif

#ifdef __LCAO
template class DiagoIterAssist<double, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class DiagoIterAssist<double, base_device::DEVICE_GPU>;
#endif
#endif
} // namespace hsolver
