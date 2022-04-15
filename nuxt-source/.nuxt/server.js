import Vue from 'vue'
import { joinURL, normalizeURL, withQuery } from 'ufo'
import fetch from 'node-fetch'
import middleware from './middleware.js'
import {
  applyAsyncData,
  middlewareSeries,
  sanitizeComponent,
  getMatchedComponents,
  promisify
} from './utils.js'
import fetchMixin from './mixins/fetch.server'
import { createApp, NuxtError } from './index.js'
import NuxtLink from './components/nuxt-link.server.js' // should be included after ./index.js

// Update serverPrefetch strategy
Vue.config.optionMergeStrategies.serverPrefetch = Vue.config.optionMergeStrategies.created

// Fetch mixin
if (!Vue.__nuxt__fetch__mixin__) {
  Vue.mixin(fetchMixin)
  Vue.__nuxt__fetch__mixin__ = true
}

// Component: <NuxtLink>
Vue.component(NuxtLink.name, NuxtLink)
Vue.component('NLink', NuxtLink)

if (!global.fetch) { global.fetch = fetch }

const noopApp = () => new Vue({ render: h => h('div', { domProps: { id: '__nuxt' } }) })

const createNext = ssrContext => (opts) => {
  // If static target, render on client-side
  ssrContext.redirected = opts
  if (ssrContext.target === 'static' || !ssrContext.res) {
    ssrContext.nuxt.serverRendered = false
    return
  }
  let fullPath = withQuery(opts.path, opts.query)
  const $config = ssrContext.runtimeConfig || {}
  const routerBase = ($config._app && $config._app.basePath) || '/LaTeX-OCR/'
  if (!fullPath.startsWith('http') && (routerBase !== '/' && !fullPath.startsWith(routerBase))) {
    fullPath = joinURL(routerBase, fullPath)
  }
  // Avoid loop redirect
  if (decodeURI(fullPath) === decodeURI(ssrContext.url)) {
    ssrContext.redirected = false
    return
  }
  ssrContext.res.writeHead(opts.status, {
    Location: normalizeURL(fullPath)
  })
  ssrContext.res.end()
}

// This exported function will be called by `bundleRenderer`.
// This is where we perform data-prefetching to determine the
// state of our application before actually rendering it.
// Since data fetching is async, this function is expected to
// return a Promise that resolves to the app instance.
export default async (ssrContext) => {
  // Create ssrContext.next for simulate next() of beforeEach() when wanted to redirect
  ssrContext.redirected = false
  ssrContext.next = createNext(ssrContext)
  // Used for beforeNuxtRender({ Components, nuxtState })
  ssrContext.beforeRenderFns = []
  // Nuxt object (window.{{globals.context}}, defaults to window.__NUXT__)
  ssrContext.nuxt = { layout: 'default', data: [], fetch: {}, error: null, serverRendered: true, routePath: '' }

    ssrContext.fetchCounters = {}

  // Remove query from url is static target

  if (ssrContext.url) {
    ssrContext.url = ssrContext.url.split('?')[0]
  }

  // Public runtime config
  ssrContext.nuxt.config = ssrContext.runtimeConfig.public
  if (ssrContext.nuxt.config._app) {
    __webpack_public_path__ = joinURL(ssrContext.nuxt.config._app.cdnURL, ssrContext.nuxt.config._app.assetsPath)
  }
  // Create the app definition and the instance (created for each request)
  const { app, router } = await createApp(ssrContext, ssrContext.runtimeConfig.private)
  const _app = new Vue(app)
  // Add ssr route path to nuxt context so we can account for page navigation between ssr and csr
  ssrContext.nuxt.routePath = app.context.route.path

  // Add meta infos (used in renderer.js)
  ssrContext.meta = _app.$meta()

  // Keep asyncData for each matched component in ssrContext (used in app/utils.js via this.$ssrContext)
  ssrContext.asyncData = {}

  const beforeRender = async () => {
    // Call beforeNuxtRender() methods
    await Promise.all(ssrContext.beforeRenderFns.map(fn => promisify(fn, { Components, nuxtState: ssrContext.nuxt })))
  }

  const renderErrorPage = async () => {
    // Don't server-render the page in static target
    if (ssrContext.target === 'static') {
      ssrContext.nuxt.serverRendered = false
    }

    // Load layout for error page
    const layout = (NuxtError.options || NuxtError).layout
    const errLayout = typeof layout === 'function' ? layout.call(NuxtError, app.context) : layout
    ssrContext.nuxt.layout = errLayout || 'default'
    await _app.loadLayout(errLayout)
    _app.setLayout(errLayout)

    await beforeRender()
    return _app
  }
  const render404Page = () => {
    app.context.error({ statusCode: 404, path: ssrContext.url, message: 'This page could not be found' })
    return renderErrorPage()
  }

  // Components are already resolved by setContext -> getRouteData (app/utils.js)
  const Components = getMatchedComponents(app.context.route)

  /*
  ** Call global middleware (nuxt.config.js)
  */
  let midd = []
  midd = midd.map((name) => {
    if (typeof name === 'function') {
      return name
    }
    if (typeof middleware[name] !== 'function') {
      app.context.error({ statusCode: 500, message: 'Unknown middleware ' + name })
    }
    return middleware[name]
  })
  await middlewareSeries(midd, app.context)
  // ...If there is a redirect or an error, stop the process
  if (ssrContext.redirected) {
    return noopApp()
  }
  if (ssrContext.nuxt.error) {
    return renderErrorPage()
  }

  /*
  ** Set layout
  */
  let layout = Components.length ? Components[0].options.layout : NuxtError.layout
  if (typeof layout === 'function') {
    layout = layout(app.context)
  }
  await _app.loadLayout(layout)
  if (ssrContext.nuxt.error) {
    return renderErrorPage()
  }
  layout = _app.setLayout(layout)
  ssrContext.nuxt.layout = _app.layoutName

  /*
  ** Call middleware (layout + pages)
  */
  midd = []

  layout = sanitizeComponent(layout)
  if (layout.options.middleware) {
    midd = midd.concat(layout.options.middleware)
  }

  Components.forEach((Component) => {
    if (Component.options.middleware) {
      midd = midd.concat(Component.options.middleware)
    }
  })
  midd = midd.map((name) => {
    if (typeof name === 'function') {
      return name
    }
    if (typeof middleware[name] !== 'function') {
      app.context.error({ statusCode: 500, message: 'Unknown middleware ' + name })
    }
    return middleware[name]
  })
  await middlewareSeries(midd, app.context)
  // ...If there is a redirect or an error, stop the process
  if (ssrContext.redirected) {
    return noopApp()
  }
  if (ssrContext.nuxt.error) {
    return renderErrorPage()
  }

  /*
  ** Call .validate()
  */
  let isValid = true
  try {
    for (const Component of Components) {
      if (typeof Component.options.validate !== 'function') {
        continue
      }

      isValid = await Component.options.validate(app.context)

      if (!isValid) {
        break
      }
    }
  } catch (validationError) {
    // ...If .validate() threw an error
    app.context.error({
      statusCode: validationError.statusCode || '500',
      message: validationError.message
    })
    return renderErrorPage()
  }

  // ...If .validate() returned false
  if (!isValid) {
    // Render a 404 error page
    return render404Page()
  }

  // If no Components found, returns 404
  if (!Components.length) {
    return render404Page()
  }

  // Call asyncData & fetch hooks on components matched by the route.
  const asyncDatas = await Promise.all(Components.map((Component) => {
    const promises = []

    // Call asyncData(context)
    if (Component.options.asyncData && typeof Component.options.asyncData === 'function') {
      const promise = promisify(Component.options.asyncData, app.context)
      promise.then((asyncDataResult) => {
        ssrContext.asyncData[Component.cid] = asyncDataResult
        applyAsyncData(Component)
        return asyncDataResult
      })
      promises.push(promise)
    } else {
      promises.push(null)
    }

    // Call fetch(context)
    if (Component.options.fetch && Component.options.fetch.length) {
      promises.push(Component.options.fetch(app.context))
    } else {
      promises.push(null)
    }

    return Promise.all(promises)
  }))

  // datas are the first row of each
  ssrContext.nuxt.data = asyncDatas.map(r => r[0] || {})

  // ...If there is a redirect or an error, stop the process
  if (ssrContext.redirected) {
    return noopApp()
  }
  if (ssrContext.nuxt.error) {
    return renderErrorPage()
  }

  // Call beforeNuxtRender methods & add store state
  await beforeRender()

  return _app
}
