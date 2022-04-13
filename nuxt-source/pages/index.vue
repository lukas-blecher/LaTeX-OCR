<template>
  <v-sheet flat dark min-height="500" class="px-7 py-7">
    <v-card-title>Overview</v-card-title>
    <v-img
      class="mx-auto"
      src="https://user-images.githubusercontent.com/55287601/109183599-69431f00-778e-11eb-9809-d42b9451e018.png"
      :width="650"
      contain
      :aspect-ratio="649 / 319"
    ></v-img>
    Convert an image containing a formula to corresponding LaTeX code. <br />
    There is a User Interface written with
    <a href="https://www.riverbankcomputing.com/software/pyqt/">PyQt5</a>
    <v-img
      class="mx-auto"
      src="https://user-images.githubusercontent.com/55287601/117812740-77b7b780-b262-11eb-81f6-fc19766ae2ae.gif"
      :width="650"
      contain
    ></v-img>
    See <NuxtLink to="performance">here</NuxtLink> for performance information.
    <v-card-title id="Install">Install</v-card-title>
    As of right now you have to use a Python environment with
    <a href="https://pytorch.org">PyTorch</a> to use the model. <br />
    Easy installation:
    <div class="pl-5 py-4">
      <code>pip install pix2tex</code>
    </div>
    For more information go to the
    <v-btn
      icon
      :ripple="false"
      x-small
      href="https://github.com/lukas-blecher/LaTeX-OCR"
    >
      <v-icon small>mdi-github </v-icon>
    </v-btn>
    <a class="ml-n1" href="https://github.com/lukas-blecher/LaTeX-OCR"
      >GitHub repository</a
    >.<br />A standalone version is planned. For Linux an AppImage is in
    development (<a href="https://github.com/lukas-blecher/LaTeX-OCR/pull/120"
      >link</a
    >).
    <v-card-title>Troubleshooting</v-card-title>
    If you run into problems, please see if you can find the solution in the
    <a href="https://github.com/lukas-blecher/LaTeX-OCR/issues?q=is%3Aissue"
      >issues</a
    >-tab. If not open a new issue.
    <v-card-title>Contributing</v-card-title>
    I would love some help for further development of the project. If you want
    to contribute, there are many areas that need to be improved.
    <v-list flat two-line>
      <v-list-item v-for="(item, i) in items" :key="i">
        <v-list-item-icon class="my-auto">
          <v-icon v-text="item.icon"></v-icon>
        </v-list-item-icon>
        <v-list-item-content>
          <v-list-item-title v-text="item.text"></v-list-item-title>
          <v-list-item-subtitle v-text="item.subtitle"></v-list-item-subtitle>
        </v-list-item-content>
      </v-list-item>
    </v-list>
    If you want to help with any of the above or any other point, feel free to
    fork the repository and submit a pull request.
  </v-sheet>
</template>

<script>
export default {
  data: () => ({
    items: [
      {
        text: 'Handwritten formulae',
        subtitle:
          'Use CROHME, Im2Latex-Handwritten and maybe https://www.kaggle.com/aidapearson/ocr-data',
        icon: 'mdi-checkbox-blank-outline',
      },
      {
        text: 'Find better hyperparameters',
        subtitle:
          "I haven't tried out many sets of hyperparameters. Could potentially improve performance drastically.",
        icon: 'mdi-checkbox-blank-outline',
      },
      {
        text: 'Tweak model structure',
        subtitle: 'Attention tricks, backbone model etc.',
        icon: 'mdi-checkbox-marked-outline',
      },
      {
        text: 'Distillation',
        subtitle: 'Introduce more of the DeiT approach.',
        icon: 'mdi-checkbox-blank-outline',
      },
      {
        text: 'Data scraping',
        subtitle: 'Fix equation extraction (regex) and collect more data.',
        icon: 'mdi-checkbox-blank-outline',
      },
      {
        text: 'Trace model',
        subtitle:
          'Using torchscript or ONNX. Useful for standalone application.',
        icon: 'mdi-checkbox-blank-outline',
      },
      {
        text: 'Desktop application',
        subtitle: 'Never done anything with Qt. Help wanted!',
        icon: 'mdi-checkbox-blank-outline',
      },
    ],
  }),
}
</script>
