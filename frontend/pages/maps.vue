<template>
  <div class="min-h-screen bg-gray-50">
    <!-- Navigation Header -->
    <nav class="bg-white shadow-sm border-b">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between h-16">
          <div class="flex items-center">
            <div class="flex-shrink-0">
              <NuxtLink to="/" class="text-xl font-bold text-gray-900">
                🚗 Auto Thievia
              </NuxtLink>
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
      <div class="px-4 py-6 sm:px-0">
        <!-- Page Header -->
        <div class="mb-8">
          <h1 class="text-3xl font-bold text-gray-900">Interactive Maps</h1>
          <p class="mt-2 text-gray-600">Generate and view interactive maps for auto theft investigation</p>
        </div>

        <!-- Map Generation Controls -->
        <div class="card mb-8">
          <h2 class="text-xl font-semibold text-gray-900 mb-6">Generate New Map</h2>
          
          <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
            <!-- Theft Analysis Map -->
            <div class="border rounded-lg p-6 hover:bg-gray-50">
              <div class="text-center">
                <div class="w-12 h-12 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  🚨
                </div>
                <h3 class="text-lg font-medium text-gray-900 mb-2">Theft Analysis</h3>
                <p class="text-sm text-gray-600 mb-4">
                  Map auto theft incidents with hotspot analysis
                </p>
                <button @click="generateTheftMap" :disabled="loading" class="btn-primary w-full">
                  <span v-if="loading && activeGeneration === 'theft'" class="flex items-center justify-center">
                    <div class="loading-spinner w-4 h-4 mr-2"></div>
                    Generating...
                  </span>
                  <span v-else>Generate Theft Map</span>
                </button>
              </div>
            </div>

            <!-- Suspect Analysis Map -->
            <div class="border rounded-lg p-6 hover:bg-gray-50">
              <div class="text-center">
                <div class="w-12 h-12 bg-yellow-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  👤
                </div>
                <h3 class="text-lg font-medium text-gray-900 mb-2">Suspect Analysis</h3>
                <p class="text-sm text-gray-600 mb-4">
                  Map suspect locations and risk assessments
                </p>
                <button @click="generateSuspectMap" :disabled="loading" class="btn-primary w-full">
                  <span v-if="loading && activeGeneration === 'suspect'" class="flex items-center justify-center">
                    <div class="loading-spinner w-4 h-4 mr-2"></div>
                    Generating...
                  </span>
                  <span v-else>Generate Suspect Map</span>
                </button>
              </div>
            </div>

            <!-- Recovery Analysis Map -->
            <div class="border rounded-lg p-6 hover:bg-gray-50">
              <div class="text-center">
                <div class="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  🔄
                </div>
                <h3 class="text-lg font-medium text-gray-900 mb-2">Recovery Analysis</h3>
                <p class="text-sm text-gray-600 mb-4">
                  Map vehicle recovery locations and patterns
                </p>
                <button @click="generateRecoveryMap" :disabled="loading" class="btn-primary w-full">
                  <span v-if="loading && activeGeneration === 'recovery'" class="flex items-center justify-center">
                    <div class="loading-spinner w-4 h-4 mr-2"></div>
                    Generating...
                  </span>
                  <span v-else>Generate Recovery Map</span>
                </button>
              </div>
            </div>
          </div>
        </div>

        <!-- Success Message -->
        <div v-if="successMessage" class="bg-green-50 border border-green-200 rounded-md p-4 mb-6">
          <div class="flex">
            <div class="flex-shrink-0">
              <svg class="h-5 w-5 text-green-400" viewBox="0 0 20 20" fill="currentColor">
                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd" />
              </svg>
            </div>
            <div class="ml-3">
              <p class="text-sm font-medium text-green-800">
                {{ successMessage }}
              </p>
            </div>
          </div>
        </div>

        <!-- Error Message -->
        <div v-if="errorMessage" class="bg-red-50 border border-red-200 rounded-md p-4 mb-6">
          <div class="flex">
            <div class="flex-shrink-0">
              <svg class="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" />
              </svg>
            </div>
            <div class="ml-3">
              <p class="text-sm font-medium text-red-800">
                {{ errorMessage }}
              </p>
            </div>
          </div>
        </div>

        <!-- Available Maps -->
        <div class="card">
          <div class="flex justify-between items-center mb-6">
            <h2 class="text-xl font-semibold text-gray-900">Available Maps</h2>
            <button @click="refreshMaps" class="btn-secondary">
              <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
              </svg>
              Refresh
            </button>
          </div>

          <div v-if="mapsLoading" class="flex justify-center py-12">
            <div class="loading-spinner"></div>
          </div>

          <div v-else-if="maps.length === 0" class="text-center py-12 text-gray-500">
            <svg class="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
            </svg>
            <h3 class="mt-2 text-sm font-medium text-gray-900">No maps yet</h3>
            <p class="mt-1 text-sm text-gray-500">Get started by generating your first map above.</p>
          </div>

          <div v-else class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div v-for="map in maps" :key="map.map_id" class="border rounded-lg p-6 hover:shadow-md transition-shadow">
              <div class="flex items-start justify-between mb-3">
                <h3 class="text-lg font-medium text-gray-900 truncate">
                  {{ map.filename.replace('.html', '') }}
                </h3>
                <span :class="getMapTypeClass(map.map_id)">
                  {{ getMapType(map.map_id) }}
                </span>
              </div>
              
              <div class="space-y-2 text-sm text-gray-600 mb-4">
                <p>Created: {{ formatDate(map.created_at) }}</p>
                <p>Size: {{ map.size_kb }} KB</p>
              </div>
              
              <div class="flex space-x-2">
                <a :href="apiBase + map.view_url" target="_blank" class="btn-primary flex-1 text-center">
                  View Map
                </a>
                <button @click="copyMapUrl(map)" class="btn-secondary">
                  📋
                </button>
              </div>
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
  title: 'Maps - Auto Thievia'
})

// Composables
const config = useRuntimeConfig()
const apiBase = config.public.apiBase

// Reactive data
const loading = ref(false)
const mapsLoading = ref(true)
const activeGeneration = ref('')
const maps = ref([])
const successMessage = ref('')
const errorMessage = ref('')

// Methods
const clearMessages = () => {
  successMessage.value = ''
  errorMessage.value = ''
}

const showSuccess = (message) => {
  clearMessages()
  successMessage.value = message
  setTimeout(() => successMessage.value = '', 5000)
}

const showError = (message) => {
  clearMessages()
  errorMessage.value = message
  setTimeout(() => errorMessage.value = '', 5000)
}

const generateTheftMap = async () => {
  try {
    loading.value = true
    activeGeneration.value = 'theft'
    clearMessages()
    
    const response = await $fetch(`${apiBase}/maps/theft`, {
      method: 'GET',
      query: {
        use_sample: true,
        center_lat: 40.7357,
        center_lon: -74.1723,
        zoom_start: 12
      }
    })
    
    showSuccess(`Theft map generated successfully! ${response.point_count} incidents mapped.`)
    await refreshMaps()
    
  } catch (error) {
    console.error('Error generating theft map:', error)
    showError('Failed to generate theft map. Please try again.')
  } finally {
    loading.value = false
    activeGeneration.value = ''
  }
}

const generateSuspectMap = async () => {
  try {
    loading.value = true
    activeGeneration.value = 'suspect'
    clearMessages()
    
    const response = await $fetch(`${apiBase}/maps/suspects`, {
      method: 'GET',
      query: {
        risk_levels: 'High,Critical',
        include_arrests: true,
        zoom_start: 12
      }
    })
    
    showSuccess(`Suspect map generated successfully! ${response.point_count} suspects mapped.`)
    await refreshMaps()
    
  } catch (error) {
    console.error('Error generating suspect map:', error)
    showError('Failed to generate suspect map. Please try again.')
  } finally {
    loading.value = false
    activeGeneration.value = ''
  }
}

const generateRecoveryMap = async () => {
  try {
    loading.value = true
    activeGeneration.value = 'recovery'
    clearMessages()
    
    const response = await $fetch(`${apiBase}/maps/recovery`, {
      method: 'GET',
      query: {
        zoom_start: 12,
        include_criminal_locations: true
      }
    })
    
    showSuccess(`Recovery map generated successfully! ${response.point_count} recoveries mapped.`)
    await refreshMaps()
    
  } catch (error) {
    console.error('Error generating recovery map:', error)
    showError('Failed to generate recovery map. Please try again.')
  } finally {
    loading.value = false
    activeGeneration.value = ''
  }
}

const refreshMaps = async () => {
  try {
    mapsLoading.value = true
    const response = await $fetch(`${apiBase}/maps/list`)
    maps.value = response.maps
  } catch (error) {
    console.error('Error fetching maps:', error)
    showError('Failed to load maps list.')
  } finally {
    mapsLoading.value = false
  }
}

const formatDate = (dateString) => {
  return new Date(dateString).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  })
}

const getMapType = (mapId) => {
  if (mapId.includes('theft')) return 'Theft'
  if (mapId.includes('suspect')) return 'Suspect'
  if (mapId.includes('recovery')) return 'Recovery'
  return 'Custom'
}

const getMapTypeClass = (mapId) => {
  const baseClass = 'status-indicator text-xs'
  if (mapId.includes('theft')) return `${baseClass} bg-red-100 text-red-800`
  if (mapId.includes('suspect')) return `${baseClass} bg-yellow-100 text-yellow-800`
  if (mapId.includes('recovery')) return `${baseClass} bg-green-100 text-green-800`
  return `${baseClass} bg-blue-100 text-blue-800`
}

const copyMapUrl = async (map) => {
  try {
    const url = `${apiBase}${map.view_url}`
    await navigator.clipboard.writeText(url)
    showSuccess('Map URL copied to clipboard!')
  } catch (error) {
    console.error('Failed to copy URL:', error)
    showError('Failed to copy URL to clipboard.')
  }
}

// Lifecycle
onMounted(() => {
  refreshMaps()
})
</script>
