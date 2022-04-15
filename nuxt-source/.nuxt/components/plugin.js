import Vue from 'vue'
import * as components from './index'

for (const name in components) {
  Vue.component(name, components[name])
  Vue.component('Lazy' + name, components[name])
}
