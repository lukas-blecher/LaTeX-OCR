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
    <a class="ml-n1" href="https://github.com/lukas-blecher/LaTeX-OCR"
      >GitHub repository</a
    >.<br />A standalone version is planned.
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
        subtitle: 'Maybe use https://www.kaggle.com/aidapearson/ocr-data',
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
