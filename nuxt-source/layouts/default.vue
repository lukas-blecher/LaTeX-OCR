<template>
  <v-app dark>
    <v-app-bar :clipped-left="clipped" fixed app>
      <v-app-bar-title class="px-3"
        ><NuxtLink to="/" tag="r" style="cursor: pointer">
          LaTeX OCR
        </NuxtLink></v-app-bar-title
      >
      <NuxtLink to="/#Install" tag="r" style="cursor: pointer" class="px-3">
        Install
      </NuxtLink>
      <NuxtLink to="/performance" tag="r" style="cursor: pointer" class="px-3">
        Performance
      </NuxtLink>
      <v-spacer />
      <v-tooltip bottom>
        <template v-slot:activator="{ on, attrs }">
          <v-btn
            :ripple="false"
            class="mx-1"
            href="https://github.com/lukas-blecher/LaTeX-OCR/releases/latest"
            v-bind="attrs"
            v-on="on"
            >Download</v-btn
          >
          <v-btn
            icon
            href="https://github.com/lukas-blecher/LaTeX-OCR"
            class="mx-1"
          >
            <v-icon large>mdi-github</v-icon>
          </v-btn>
        </template>
        <span>Coming soon</span>
      </v-tooltip>
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
