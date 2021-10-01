## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/BrightGu/MediumVC/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

## MediumVC: Any2-to-any voice conversion using specific-speaker speeches as intermedium features

SingleVC performs A2O VC through a self-supervised task((X<sub>i</sub>  →X̂<sup>s</sup><sub>i</sub> → X̂<sub>i</sub>  )).  X̂<sup>s</sup><sub>i</sub> is  a PSDR-processed speech with pitch-shifted s. The more details can be access here.

We propose MediumVC, an utterance-level method towards A2A. Before that, we propose [SingleVC](https://github.com/BrightGu/SingleVC) to perform A2O tasks(X<sub>i</sub> → Ŷ<sub>i</sub>) , X<sub>i</sub> means utterance i spoken by X). The Ŷ<sub>i</sub> are considered as SSIF. To build SingleVC, we employ a novel data augment strategy: pitch-shifted and duration-remained(PSDR) to produce paired asymmetrical training data. Then, based on pre-trained SingleVC, MediumVC performs an asymmetrical reconstruction task(Ŷ<sub>i</sub> → X̂<sub>i</sub>). Due to the asymmetrical reconstruction mode, MediumVC achieves more efficient feature decoupling and fusion. Experiments demonstrate MediumVC performs strong robustness for unseen speakers across multiple public datasets.

This page provides converted speech samples. 

### VCTK
1. F2F, The Norsemen considered the rainbow as a bridge over which the gods passed from earth to their home in the sky.
2. F2M, We also need a small plastic snake and a big toy frog for the kids. 
3. M2F, Since then physicists have found that it is not reflection, but refraction by the raindrops which causes the rainbows. 
4. M2M, Many complicated ideas about the rainbow have been formed.

<table>
   <tr>
      <td>Source</td>
      <td>Target</td>
      <td>MediumVC</td>
      <td>MwuS</td>
      <td>MwoS</td>
      <td>FragmentVC</td>
      <td>AutoVC</td>
      <td>Ada-In VC</td>
   </tr>
   <tr>
      <td><audio id="audio" controls="" preload="none"> <source id="V1_s" src="converted_samples/VCTK/V1/1_src_fp240_016.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="V1_t" src="converted_samples/VCTK/V1/1_tar_fp231_018.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="V1_A" src="converted_samples/VCTK/V1/A_MediumVC.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="V1_B" src="converted_samples/VCTK/V1/B_M_WU_S.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="V1_C" src="converted_samples/VCTK/V1/C_M_WO_S.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="V1_D" src="converted_samples/VCTK/V1/D_FragmentVC.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="V1_E" src="converted_samples/VCTK/V1/E_AutoVC.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="V1_F" src="converted_samples/VCTK/V1/F_AdaIN-VC.wav"></td>
   </tr>
   <tr>
      <td><audio id="audio" controls="" preload="none"> <source id="V8_s" src="converted_samples/VCTK/V8/8_src_fp231_004.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="V8_t" src="converted_samples/VCTK/V8/8_tar_mp275_004.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="V8_A" src="converted_samples/VCTK/V8/A_MediumVC.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="V8_B" src="converted_samples/VCTK/V8/B_M_WU_S.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="V8_C" src="converted_samples/VCTK/V8/C_M_WO_S.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="V8_D" src="converted_samples/VCTK/V8/D_FragmentVC.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="V8_E" src="converted_samples/VCTK/V8/E_AutoVC.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="V8_F" src="converted_samples/VCTK/V8/F_AdaIN-VC.wav"></td>
   </tr>
   <tr>
      <td><audio id="audio" controls="" preload="none"> <source id="V14_s" src="converted_samples/VCTK/V14/14_src_mp275_019.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="V14_t" src="converted_samples/VCTK/V14/14_tar_fp231_006.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="V14_A" src="converted_samples/VCTK/V14/A_MediumVC.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="V14_B" src="converted_samples/VCTK/V14/B_M_WU_S.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="V14_C" src="converted_samples/VCTK/V14/C_M_WO_S.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="V14_D" src="converted_samples/VCTK/V14/D_FragmentVC.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="V14_E" src="converted_samples/VCTK/V14/E_AutoVC.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="V14_F" src="converted_samples/VCTK/V14/F_AdaIN-VC.wav"></td>
   </tr>
   <tr>
      <td><audio id="audio" controls="" preload="none"> <source id="V19_s" src="converted_samples/VCTK/V19/19_src_mp275_020.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="V19_t" src="converted_samples/VCTK/V19/19_tar_mp226_009.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="V19_A" src="converted_samples/VCTK/V19/A_MediumVC.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="V19_B" src="converted_samples/VCTK/V19/B_M_WU_S.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="V19_C" src="converted_samples/VCTK/V19/C_M_WO_S.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="V19_D" src="converted_samples/VCTK/V19/D_FragmentVC.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="V19_E" src="converted_samples/VCTK/V19/E_AutoVC.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="V19_F" src="converted_samples/VCTK/V19/F_AdaIN-VC.wav"></td>
   </tr>
</table>



### LibriSpeech
1. F2F, A voice on the beach under the cliff began to sing.
2. F2M, The two started to walk back along the road toward town.
3. M2F, This was impregnable, and admitted of neither objection nor restriction.
4. M2M, From a cupboard he pulled out one of his old shirts, which he tore in pieces.

<table>
   <tr>
      <td>Source</td>
      <td>Target</td>
      <td>MediumVC</td>
      <td>MwuS</td>
      <td>MwoS</td>
      <td>FragmentVC</td>
      <td>AutoVC</td>
      <td>Ada-In VC</td>
   </tr>
   <tr>
      <td><audio id="audio" controls="" preload="none"> <source id="L2_s" src="converted_samples/LibriSpeech/L2/2_src_f1093_132891_000015_000000.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="L2_t" src="converted_samples/LibriSpeech/L2/2_tar_f1060_134451_000005_000002.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="L2_A" src="converted_samples/LibriSpeech/L2/A_MediumVC.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="L2_B" src="converted_samples/LibriSpeech/L2/B_M_WU_S.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="L2_C" src="converted_samples/LibriSpeech/L2/C_M_WO_S.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="L2_D" src="converted_samples/LibriSpeech/L2/D_FragmentVC.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="L2_E" src="converted_samples/LibriSpeech/L2/E_AutoVC.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="L2_F" src="converted_samples/LibriSpeech/L2/F_AdaIN-VC.wav"></td>
   </tr>
   <tr>
      <td><audio id="audio" controls="" preload="none"> <source id="L3_s"  src="converted_samples/LibriSpeech/L3/3_src_f1060_134451_000011_000002.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="L3_t" src="converted_samples/LibriSpeech/L3/3_tar_m1025_92814_000029_000002.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="L3_A" src="converted_samples/LibriSpeech/L3/A_MediumVC.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="L3_B" src="converted_samples/LibriSpeech/L3/B_M_WU_S.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="L3_C" src="converted_samples/LibriSpeech/L3/C_M_WO_S.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="L3_D" src="converted_samples/LibriSpeech/L3/D_FragmentVC.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="L3_E" src="converted_samples/LibriSpeech/L3/E_AutoVC.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="L3_F" src="converted_samples/LibriSpeech/L3/F_AdaIN-VC.wav"></td>
   </tr>
   <tr>
      <td><audio id="audio" controls="" preload="none"> <source id="L8_s"  src="converted_samples/LibriSpeech/L8/8_src_m1365_134830_000055_000001.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="L8_t" src="converted_samples/LibriSpeech/L8/8_tar_f1060_134451_000017_000006.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="L8_A" src="converted_samples/LibriSpeech/L8/A_MediumVC.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="L8_B" src="converted_samples/LibriSpeech/L8/B_M_WU_S.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="L8_C" src="converted_samples/LibriSpeech/L8/C_M_WO_S.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="L8_D" src="converted_samples/LibriSpeech/L8/D_FragmentVC.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="L8_E" src="converted_samples/LibriSpeech/L8/E_AutoVC.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="L8_F" src="converted_samples/LibriSpeech/L8/F_AdaIN-VC.wav"></td>
   </tr>
   <tr>
      <td><audio id="audio" controls="" preload="none"> <source id="L11_s"  src="converted_samples/LibriSpeech/L11/11_src_m1365_134830_000029_000001.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="L11_t" src="converted_samples/LibriSpeech/L11/11_tar_m1313_135020_000031_000000.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="L11_A" src="converted_samples/LibriSpeech/L11/A_MediumVC.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="L11_B" src="converted_samples/LibriSpeech/L11/B_M_WU_S.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="L11_C" src="converted_samples/LibriSpeech/L11/C_M_WO_S.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="L11_D" src="converted_samples/LibriSpeech/L11/D_FragmentVC.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="L11_E" src="converted_samples/LibriSpeech/L11/E_AutoVC.wav"></td>
      <td><audio id="audio" controls="" preload="none"> <source id="L11_F" src="converted_samples/LibriSpeech/L11/F_AdaIN-VC.wav"></td>
   </tr>
</table>

### VCC2020
1. F1, If not, it's about time somebody did.
2. F2, The figures are adjusted for seasonal variation.
3. M1, The trip was a disaster.
4. M2, Sometimes, it helps to take a step back.


