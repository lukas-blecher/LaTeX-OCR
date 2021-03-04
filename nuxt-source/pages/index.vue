<template>
  <v-sheet flat dark min-height="500" class="px-3">
    <v-card-title>Overview</v-card-title>
    <v-img
      class="mx-auto"
      src="https://user-images.githubusercontent.com/55287601/109183599-69431f00-778e-11eb-9809-d42b9451e018.png"
      :width="650"
      contain
      :aspect-ratio="649 / 319"
    ></v-img>
    <v-card-title>Features</v-card-title>
    Convert an image containing a formula to corresponding LaTeX code.
    <v-card-title>Download</v-card-title>
    As of right now you have to use a Python environment with
    <a href="https://pytorch.org">PyTorch</a> to use the model. For more
    information follow the instructions on the
    <v-btn
      icon
      :ripple="false"
      x-small
      href="https://github.com/lukas-blecher/LaTeX-OCR"
    >
      <v-icon small>mdi-github </v-icon>
    </v-btn>
    <a href="https://github.com/lukas-blecher/LaTeX-OCR">GitHub repository</a
    >.<br />A standalone version is planned.
    <v-card-title>Troubleshooting</v-card-title>
    If you run into problems, please see if you can find the solution in the
    <a href="https://github.com/lukas-blecher/LaTeX-OCR/issues?q=is%3Aissue"
      >issues</a
    >-tab. If not open a new issue.
  </v-sheet>
</template>

<script>
export default {}
</script>
