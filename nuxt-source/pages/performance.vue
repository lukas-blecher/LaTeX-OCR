<template>
  <v-sheet flat dark min-height="500" class="px-3">
    <v-card-title>Examples</v-card-title>
    Here are some examples that demonstrate the performance of the model. I
    didn't just want to optimize the metrics, that can be found in the
    literature if that means images found in the wild will not be recognized.
    <v-row class="py-2" v-for="(item, i) in latex_code" :key="i">
      <v-col cols="5">
        <v-img class="mx-auto" :src="images[i]" :height="100" contain></v-img>
      </v-col>
      <v-col cols="6">{{ item }} </v-col>
      <v-col cols="1">
        <v-icon :color="color[icon[i]]" v-html="icons[icon[i]]"></v-icon>
      </v-col>
    </v-row>
    Where
    <v-icon :color="color[1]" v-html="icons[1]"></v-icon> means the code
    generates the same image,
    <v-icon :color="color[2]" v-html="icons[2]"></v-icon> means that the
    predicted code compiles and the image is close to the original.
    <v-icon :color="color[0]" v-html="icons[0]"></v-icon> means that there are
    significant differences between the original and the predicted code.
    <v-card-title>Metrics</v-card-title>
    <table class="tg">
      <thead>
        <tr>
          <th class="tg-0kls"></th>
          <th class="tg-0kls">BLEU</th>
          <th class="tg-0kls">normed edit distance</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="tg-0kls">Performance</td>
          <td class="tg-0kls">0.88</td>
          <td class="tg-0kls">0.10</td>
        </tr>
      </tbody>
    </table>
    <br />
  </v-sheet>
</template>

<script>
export default {
  data: () => ({
    images: [
      'https://user-images.githubusercontent.com/55287601/118541376-e361bf00-b751-11eb-9f6a-d672f7671d60.png',
      'https://user-images.githubusercontent.com/55287601/118545805-60dbfe00-b757-11eb-9841-24ce8e152870.png',
      'https://user-images.githubusercontent.com/55287601/118545995-a6003000-b757-11eb-93d8-52e012a9c25c.png',
      'https://user-images.githubusercontent.com/55287601/118546404-3179c100-b758-11eb-8b30-b700daa17dae.png',
      'https://user-images.githubusercontent.com/55287601/118547279-6d615600-b759-11eb-8520-16edf78c7554.png',
      'https://user-images.githubusercontent.com/55287601/118550840-7e13cb00-b75d-11eb-954a-965bfd657364.png',
    ],
    latex_code: [
      '\\nabla\\times{\\vec{\\mathbf{B}}} - {\\frac{1}{C}} {\\frac{\\delta{\\vec{\\mathbf{E}}}}{\\delta t}}=4\\pi\\rho',
      '\\ln f_{a,\\sigma^{2}}(\\xi_{1})= \\frac{(\\xi_{1} - a)}{\\sigma^{2}}',
      'H(Y|X)=\\sum_{x\\in{\\mathcal{X}},y\\in{\\mathcal{Y}}}p(x,y) \\log\\left({\\frac{p(x)}{p(x,y)}}\\right)',
      '\\sigma=\\frac{\\pi\\alpha^{2}\\beta^{3}}{3s}\\left[1 + \\frac{s( - 1 + 4s_{W}^{2})}{2c_{W}^{2}(s - M_{Z}^{2})} + \\frac{s^{2}( - 1 + 4s_{W}^{2} + 8s_{W}^{4})}{4c_{W}^{4}(s - M_{Z}^{2})^{2}}\\right]\\;,',
      '\\begin{array}{l}{{\\mathrm{d}C= - v(r)r{\\mathrm{d}}\\varphi + v(r + {\\mathrm{d}}r)\\cdot(r + {\\mathrm{d}}r){\\mathrm{d}}\\varphi}} \\\\ {{\\mathrm{~}= - v(r)r{\\mathrm{d}}\\varphi + \\left(v(r) + {\\frac{\\partial v}{\\partial r}}{\\mathrm{d}}r\\right)(r + {\\mathrm{d}}r){\\mathrm{d}}\\varphi}} \\\\ {{\\mathrm{~}}} \\\\ {{\\mathrm{~}=v(r){\\mathrm{d}}r{\\mathrm{~}}\\varphi + {\\frac{\\partial v}{\\partial r}}r{\\mathrm{d}}r{\\mathrm{d}}{\\mathrm{~}}{\\mathrm{~}}\\varphi + {\\mathrm{O}}({\\mathrm{d}}^{3})}}\\end{array}',
      '\\begin{array}{c c}{{\\bar{u}_{s}(\\vec{p})\\cdot u_{r}(\\vec{p}) = 2m\\delta_{s r},}} & {{\\qquad u_{s}^{\\dagger}(\\vec{p})\\cdot u_{r}(\\vec{p}) = 2p^{0}\\delta_{s r},}}\\\\ {{\\bar{v}_{s}(\\vec{p})\\cdot v_{r}(\\vec{p}) = -2m\\delta_{s r},}} & {{\\qquad v_{s}^{\\dagger}(\\vec{p})\\cdot v_{r}(-\\vec{p}) = 2p^{0}\\delta_{s r},}}\\\\ {{\\bar{u}_{s}(\\vec{p})\\cdot v_{r}(\\vec{p}) = 0,}} & {{u_{s}^{\\dagger}(\\vec{p})\\cdot v_{r}(-\\vec{p}) = 0,}}\\end{array}',
    ],
    icon: [1, 1, 1, 1, 2, 2],
    icons: [
      'mdi-close-box-outline',
      'mdi-checkbox-marked-outline',
      'mdi-approximately-equal-box',
    ],
    color: ['red', 'green', 'orange'],
  }),
}
</script>
<style type="text/css">
.tg {
  border-collapse: collapse;
  border-spacing: 0;
}
.tg td {
  border-style: solid;
  border-width: 1px;
  font-family: Arial, sans-serif;
  font-size: 14px;
  overflow: hidden;
  padding: 10px 5px;
  word-break: normal;
}
.tg th {
  border-style: solid;
  border-width: 1px;
  font-family: Arial, sans-serif;
  font-size: 14px;
  font-weight: normal;
  overflow: hidden;
  padding: 10px 5px;
  word-break: normal;
}
.tg .tg-0pky {
  border-color: inherit;
  text-align: left;
  vertical-align: top;
}
</style>
