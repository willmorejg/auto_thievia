// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
    devtools: { enabled: true },

    // Global CSS
    css: ['~/assets/css/main.css'],

    // Modules
    modules: [
        '@nuxtjs/tailwindcss'
    ],

    // Runtime config
    runtimeConfig: {
        public: {
            apiBase: process.env.API_BASE_URL || 'http://localhost:8000'
        }
    },

    // App configuration
    app: {
        head: {
            title: 'Auto Thievia - GIS Analysis Dashboard',
            htmlAttrs: {
                lang: 'en'
            },
            meta: [
                { charset: 'utf-8' },
                { name: 'viewport', content: 'width=device-width, initial-scale=1' },
                { hid: 'description', name: 'description', content: 'Auto theft investigation mapping and analysis dashboard' }
            ],
            link: [
                { rel: 'icon', type: 'image/x-icon', href: '/favicon.ico' },
                { rel: 'stylesheet', href: 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css' }
            ]
        }
    },

    // Development server
    devServer: {
        port: 3000
    },

    // Build configuration
    build: {
        transpile: ['leaflet']
    },

    // SSR configuration
    ssr: false // For better client-side map rendering
})
