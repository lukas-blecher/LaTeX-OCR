<template>
  <v-app dark>
    <v-app-bar :clipped-left="clipped" fixed app>
      <v-app-bar-title class="mx-3">
        LaTeX <v-icon large class="my-auto">mdi-ocr</v-icon>
      </v-app-bar-title>
      <v-spacer />
      <v-btn
        :ripple="false"
        class="mx-1"
        href="https://github.com/lukas-blecher/LaTeX-OCR/releases/latest"
        >Download</v-btn
      >
      <v-btn
        icon
        href="https://github.com/lukas-blecher/LaTeX-OCR"
        class="mx-1"
      >
        <v-icon large>mdi-github</v-icon>
      </v-btn>
    </v-app-bar>
    <v-main>
      <v-container>
        <nuxt />
      </v-container>
    </v-main>
    <v-footer :absolute="!fixed" app>
      <span>&copy; {{ new Date().getFullYear() }} Lukas Blecher</span>
    </v-footer>
  </v-app>
</template>

<script>
export default {
  data() {
    return {
      clipped: false,
      fixed: false,
    }
  },
}
</script>
