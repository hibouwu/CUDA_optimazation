#pragma once

// One active output warp pair stores64 fragment registers per thread.
// Offsets follow the m64n128 accumulator mapping. The n256 fragment uses two
// calls selected by chunk_n; this binds one SMEM base instead of hoisting64 addresses.
__device__ __forceinline__ void r15_store_fragment64(unsigned base, const float* d) {
  asm volatile(
      "st.volatile.shared.b32 [%64+0],%0;\n"
      "st.volatile.shared.b32 [%64+4],%1;\n"
      "st.volatile.shared.b32 [%64+4096],%2;\n"
      "st.volatile.shared.b32 [%64+4100],%3;\n"
      "st.volatile.shared.b32 [%64+32],%4;\n"
      "st.volatile.shared.b32 [%64+36],%5;\n"
      "st.volatile.shared.b32 [%64+4128],%6;\n"
      "st.volatile.shared.b32 [%64+4132],%7;\n"
      "st.volatile.shared.b32 [%64+64],%8;\n"
      "st.volatile.shared.b32 [%64+68],%9;\n"
      "st.volatile.shared.b32 [%64+4160],%10;\n"
      "st.volatile.shared.b32 [%64+4164],%11;\n"
      "st.volatile.shared.b32 [%64+96],%12;\n"
      "st.volatile.shared.b32 [%64+100],%13;\n"
      "st.volatile.shared.b32 [%64+4192],%14;\n"
      "st.volatile.shared.b32 [%64+4196],%15;\n"
      "st.volatile.shared.b32 [%64+128],%16;\n"
      "st.volatile.shared.b32 [%64+132],%17;\n"
      "st.volatile.shared.b32 [%64+4224],%18;\n"
      "st.volatile.shared.b32 [%64+4228],%19;\n"
      "st.volatile.shared.b32 [%64+160],%20;\n"
      "st.volatile.shared.b32 [%64+164],%21;\n"
      "st.volatile.shared.b32 [%64+4256],%22;\n"
      "st.volatile.shared.b32 [%64+4260],%23;\n"
      "st.volatile.shared.b32 [%64+192],%24;\n"
      "st.volatile.shared.b32 [%64+196],%25;\n"
      "st.volatile.shared.b32 [%64+4288],%26;\n"
      "st.volatile.shared.b32 [%64+4292],%27;\n"
      "st.volatile.shared.b32 [%64+224],%28;\n"
      "st.volatile.shared.b32 [%64+228],%29;\n"
      "st.volatile.shared.b32 [%64+4320],%30;\n"
      "st.volatile.shared.b32 [%64+4324],%31;\n"
      "st.volatile.shared.b32 [%64+256],%32;\n"
      "st.volatile.shared.b32 [%64+260],%33;\n"
      "st.volatile.shared.b32 [%64+4352],%34;\n"
      "st.volatile.shared.b32 [%64+4356],%35;\n"
      "st.volatile.shared.b32 [%64+288],%36;\n"
      "st.volatile.shared.b32 [%64+292],%37;\n"
      "st.volatile.shared.b32 [%64+4384],%38;\n"
      "st.volatile.shared.b32 [%64+4388],%39;\n"
      "st.volatile.shared.b32 [%64+320],%40;\n"
      "st.volatile.shared.b32 [%64+324],%41;\n"
      "st.volatile.shared.b32 [%64+4416],%42;\n"
      "st.volatile.shared.b32 [%64+4420],%43;\n"
      "st.volatile.shared.b32 [%64+352],%44;\n"
      "st.volatile.shared.b32 [%64+356],%45;\n"
      "st.volatile.shared.b32 [%64+4448],%46;\n"
      "st.volatile.shared.b32 [%64+4452],%47;\n"
      "st.volatile.shared.b32 [%64+384],%48;\n"
      "st.volatile.shared.b32 [%64+388],%49;\n"
      "st.volatile.shared.b32 [%64+4480],%50;\n"
      "st.volatile.shared.b32 [%64+4484],%51;\n"
      "st.volatile.shared.b32 [%64+416],%52;\n"
      "st.volatile.shared.b32 [%64+420],%53;\n"
      "st.volatile.shared.b32 [%64+4512],%54;\n"
      "st.volatile.shared.b32 [%64+4516],%55;\n"
      "st.volatile.shared.b32 [%64+448],%56;\n"
      "st.volatile.shared.b32 [%64+452],%57;\n"
      "st.volatile.shared.b32 [%64+4544],%58;\n"
      "st.volatile.shared.b32 [%64+4548],%59;\n"
      "st.volatile.shared.b32 [%64+480],%60;\n"
      "st.volatile.shared.b32 [%64+484],%61;\n"
      "st.volatile.shared.b32 [%64+4576],%62;\n"
      "st.volatile.shared.b32 [%64+4580],%63;\n"
      :
      : "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]), "f"(d[4]), "f"(d[5]), "f"(d[6]), "f"(d[7]),
        "f"(d[8]), "f"(d[9]), "f"(d[10]), "f"(d[11]), "f"(d[12]), "f"(d[13]), "f"(d[14]), "f"(d[15]),
        "f"(d[16]), "f"(d[17]), "f"(d[18]), "f"(d[19]), "f"(d[20]), "f"(d[21]), "f"(d[22]), "f"(d[23]),
        "f"(d[24]), "f"(d[25]), "f"(d[26]), "f"(d[27]), "f"(d[28]), "f"(d[29]), "f"(d[30]), "f"(d[31]),
        "f"(d[32]), "f"(d[33]), "f"(d[34]), "f"(d[35]), "f"(d[36]), "f"(d[37]), "f"(d[38]), "f"(d[39]),
        "f"(d[40]), "f"(d[41]), "f"(d[42]), "f"(d[43]), "f"(d[44]), "f"(d[45]), "f"(d[46]), "f"(d[47]),
        "f"(d[48]), "f"(d[49]), "f"(d[50]), "f"(d[51]), "f"(d[52]), "f"(d[53]), "f"(d[54]), "f"(d[55]),
        "f"(d[56]), "f"(d[57]), "f"(d[58]), "f"(d[59]), "f"(d[60]), "f"(d[61]), "f"(d[62]), "f"(d[63]), "r"(base)
      : "memory");
}
