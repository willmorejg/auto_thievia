<template>
  <div class="min-h-screen bg-gray-50">
    <!-- Navigation Header -->
    <nav class="bg-white shadow-sm border-b">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between h-16">
          <div class="flex items-center">
            <div class="flex-shrink-0">
              <h1 class="text-xl font-bold text-gray-900">
                🚗 Auto Thievia
              </h1>
            </div>
            <div class="hidden md:ml-6 md:flex md:space-x-8">
              <NuxtLink to="/" class="nav-link">Dashboard</NuxtLink>
              <NuxtLink to="/maps" class="nav-link">Maps</NuxtLink>
              <NuxtLink to="/analysis" class="nav-link">Analysis</NuxtLink>
            </div>
          </div>
        </div>
      </div>
    </nav>

    <!-- Main Content -->
    <main class="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
      <!-- Welcome Section -->
      <div class="px-4 py-6 sm:px-0">
        <div class="card mb-8">
          <div class="text-center">
            <h2 class="text-3xl font-bold text-gray-900 mb-4">
              Auto Theft Investigation Dashboard
            </h2>
            <p class="text-lg text-gray-600 mb-6">
              GIS-powered analysis and mapping for auto theft pattern investigation
            </p>
            <div class="flex justify-center space-x-4">
              <NuxtLink to="/maps" class="btn-primary">
                🗺️ View Maps
              </NuxtLink>
              <NuxtLink to="/analysis" class="btn-secondary">
                📊 Run Analysis
              </NuxtLink>
            </div>
          </div>
        </div>

        <!-- Quick Stats -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div class="card">
            <div class="flex items-center">
              <div class="flex-shrink-0">
                <div class="w-8 h-8 bg-red-100 rounded-full flex items-center justify-center">
                  🚨
                </div>
              </div>
              <div class="ml-5 w-0 flex-1">
                <dl>
                  <dt class="text-sm font-medium text-gray-500 truncate">
                    Active Cases
                  </dt>
                  <dd class="text-lg font-medium text-gray-900">
                    {{ stats.activeCases }}
                  </dd>
                </dl>
              </div>
            </div>
          </div>

          <div class="card">
            <div class="flex items-center">
              <div class="flex-shrink-0">
                <div class="w-8 h-8 bg-yellow-100 rounded-full flex items-center justify-center">
                  👤
                </div>
              </div>
              <div class="ml-5 w-0 flex-1">
                <dl>
                  <dt class="text-sm font-medium text-gray-500 truncate">
                    High-Risk Suspects
                  </dt>
                  <dd class="text-lg font-medium text-gray-900">
                    {{ stats.highRiskSuspects }}
                  </dd>
                </dl>
              </div>
            </div>
          </div>

          <div class="card">
            <div class="flex items-center">
              <div class="flex-shrink-0">
                <div class="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center">
                  🔄
                </div>
              </div>
              <div class="ml-5 w-0 flex-1">
                <dl>
                  <dt class="text-sm font-medium text-gray-500 truncate">
                    Recoveries This Month
                  </dt>
                  <dd class="text-lg font-medium text-gray-900">
                    {{ stats.recoveriesThisMonth }}
                  </dd>
                </dl>
              </div>
            </div>
          </div>
        </div>

        <!-- Recent Maps -->
        <div class="card mb-8">
          <h3 class="text-lg font-medium text-gray-900 mb-4">Recent Maps</h3>
          <div v-if="loading" class="flex justify-center py-8">
            <div class="loading-spinner"></div>
          </div>
          <div v-else-if="recentMaps.length === 0" class="text-center py-8 text-gray-500">
            No maps generated yet. <NuxtLink to="/maps" class="text-blue-600 hover:text-blue-800">Create your first map</NuxtLink>
          </div>
          <div v-else class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div v-for="map in recentMaps" :key="map.map_id" class="border rounded-lg p-4 hover:bg-gray-50">
              <h4 class="font-medium text-gray-900 mb-2">{{ map.filename }}</h4>
              <p class="text-sm text-gray-600 mb-3">
                Created: {{ formatDate(map.created_at) }}
              </p>
              <div class="flex justify-between items-center">
                <span class="text-xs text-gray-500">{{ map.size_kb }} KB</span>
                <a :href="apiBase + map.view_url" target="_blank" class="text-blue-600 hover:text-blue-800 text-sm">
                  View Map →
                </a>
              </div>
            </div>
          </div>
        </div>

        <!-- API Status -->
        <div class="card">
          <h3 class="text-lg font-medium text-gray-900 mb-4">System Status</h3>
          <div class="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div v-for="(status, service) in systemStatus" :key="service" class="text-center">
              <div :class="['status-indicator', status === 'available' ? 'status-success' : 'status-error']">
                {{ status }}
              </div>
              <p class="text-xs text-gray-600 mt-1 capitalize">{{ service.replace('_', ' ') }}</p>
            </div>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script setup>
// Page meta
definePageMeta({
  title: 'Dashboard - Auto Thievia'
})

// Composables
const config = useRuntimeConfig()
const apiBase = config.public.apiBase

// Reactive data
const loading = ref(true)
const stats = ref({
  activeCases: 42,
  highRiskSuspects: 15,
  recoveriesThisMonth: 8
})
const recentMaps = ref([])
const systemStatus = ref({})

// Methods
const formatDate = (dateString) => {
  return new Date(dateString).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  })
}

const fetchData = async () => {
  try {
    loading.value = true
    
    // Fetch recent maps
    const mapsResponse = await $fetch(`${apiBase}/maps/list`)
    recentMaps.value = mapsResponse.maps.slice(0, 6) // Show latest 6 maps
    
    // Fetch system status
    const healthResponse = await $fetch(`${apiBase}/health`)
    systemStatus.value = healthResponse.services
    
  } catch (error) {
    console.error('Error fetching dashboard data:', error)
  } finally {
    loading.value = false
  }
}

// Lifecycle
onMounted(() => {
  fetchData()
})
</script>
