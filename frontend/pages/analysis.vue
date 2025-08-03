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
          <h1 class="text-3xl font-bold text-gray-900">Analysis & Intelligence</h1>
          <p class="mt-2 text-gray-600">Perform advanced analysis on auto theft patterns and suspect activities</p>
        </div>

        <!-- Analysis Controls -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
          <!-- Theft Pattern Analysis -->
          <div class="card">
            <h2 class="text-xl font-semibold text-gray-900 mb-4">Theft Pattern Analysis</h2>
            <p class="text-gray-600 mb-6">Analyze theft incidents to identify hotspots and patterns.</p>
            
            <div class="space-y-4 mb-6">
              <div>
                <label class="form-label">Analysis Center (Latitude)</label>
                <input v-model.number="theftAnalysis.center_lat" type="number" step="0.0001" class="form-input">
              </div>
              <div>
                <label class="form-label">Analysis Center (Longitude)</label>
                <input v-model.number="theftAnalysis.center_lon" type="number" step="0.0001" class="form-input">
              </div>
              <div>
                <label class="form-label">Analysis Radius (km)</label>
                <input v-model.number="theftAnalysis.radius_km" type="number" step="0.1" class="form-input">
              </div>
            </div>
            
            <button @click="runTheftAnalysis" :disabled="analysisLoading.theft" class="btn-primary w-full">
              <span v-if="analysisLoading.theft" class="flex items-center justify-center">
                <div class="loading-spinner w-4 h-4 mr-2"></div>
                Analyzing...
              </span>
              <span v-else>🔍 Run Theft Analysis</span>
            </button>
          </div>

          <!-- Suspect Analysis -->
          <div class="card">
            <h2 class="text-xl font-semibold text-gray-900 mb-4">Suspect Analysis</h2>
            <p class="text-gray-600 mb-6">Analyze suspect data and risk assessments.</p>
            
            <div class="space-y-4 mb-6">
              <div>
                <label class="form-label">Risk Levels</label>
                <div class="mt-2 space-y-2">
                  <label v-for="level in riskLevels" :key="level" class="inline-flex items-center mr-4">
                    <input v-model="suspectAnalysis.risk_levels" :value="level" type="checkbox" class="form-checkbox">
                    <span class="ml-2 text-sm">{{ level }}</span>
                  </label>
                </div>
              </div>
              <div>
                <label class="form-label">Days Threshold</label>
                <input v-model.number="suspectAnalysis.days_threshold" type="number" class="form-input">
              </div>
              <div class="flex items-center">
                <input v-model="suspectAnalysis.include_arrests" type="checkbox" class="form-checkbox">
                <label class="ml-2 text-sm">Include arrest data</label>
              </div>
            </div>
            
            <button @click="runSuspectAnalysis" :disabled="analysisLoading.suspect" class="btn-primary w-full">
              <span v-if="analysisLoading.suspect" class="flex items-center justify-center">
                <div class="loading-spinner w-4 h-4 mr-2"></div>
                Analyzing...
              </span>
              <span v-else>👤 Run Suspect Analysis</span>
            </button>
          </div>
        </div>

        <!-- Analysis Results -->
        <div v-if="results.theft || results.suspect" class="space-y-8">
          <!-- Theft Analysis Results -->
          <div v-if="results.theft" class="card">
            <h3 class="text-lg font-semibold text-gray-900 mb-4">Theft Analysis Results</h3>
            
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
              <div class="bg-gray-50 rounded-lg p-4">
                <div class="text-2xl font-bold text-gray-900">{{ results.theft.data_summary.total_incidents }}</div>
                <div class="text-sm text-gray-600">Total Incidents</div>
              </div>
              <div class="bg-gray-50 rounded-lg p-4">
                <div class="text-2xl font-bold text-gray-900">{{ results.theft.data_summary.radius_km }} km</div>
                <div class="text-sm text-gray-600">Analysis Radius</div>
              </div>
              <div class="bg-gray-50 rounded-lg p-4">
                <div class="text-2xl font-bold text-gray-900">
                  {{ results.theft.clustering.cluster_count || 'N/A' }}
                </div>
                <div class="text-sm text-gray-600">Clusters Found</div>
              </div>
            </div>

            <div v-if="results.theft.clustering && !results.theft.clustering.error" class="bg-blue-50 rounded-lg p-4">
              <h4 class="font-medium text-blue-900 mb-2">Clustering Analysis</h4>
              <div class="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span class="text-blue-700">Clustered Points:</span>
                  <span class="font-medium">{{ results.theft.clustering.clustered_points }}</span>
                </div>
                <div>
                  <span class="text-blue-700">Noise Points:</span>
                  <span class="font-medium">{{ results.theft.clustering.noise_points }}</span>
                </div>
              </div>
            </div>

            <div class="mt-4 text-xs text-gray-500">
              Generated: {{ formatDate(results.theft.generated_at) }}
            </div>
          </div>

          <!-- Suspect Analysis Results -->
          <div v-if="results.suspect" class="card">
            <h3 class="text-lg font-semibold text-gray-900 mb-4">Suspect Analysis Results</h3>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              <div class="bg-gray-50 rounded-lg p-4">
                <div class="text-2xl font-bold text-gray-900">{{ results.suspect.summary.total_suspects }}</div>
                <div class="text-sm text-gray-600">Total Suspects</div>
              </div>
              <div class="bg-gray-50 rounded-lg p-4">
                <div class="text-2xl font-bold text-gray-900">{{ results.suspect.summary.filtered_suspects }}</div>
                <div class="text-sm text-gray-600">Filtered Suspects</div>
              </div>
            </div>

            <div v-if="results.suspect.summary.risk_distribution" class="bg-yellow-50 rounded-lg p-4 mb-4">
              <h4 class="font-medium text-yellow-900 mb-2">Risk Distribution</h4>
              <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div v-for="(count, risk) in results.suspect.summary.risk_distribution" :key="risk">
                  <span class="text-yellow-700">{{ risk }}:</span>
                  <span class="font-medium">{{ count }}</span>
                </div>
              </div>
            </div>

            <div class="bg-gray-100 rounded-lg p-4">
              <h4 class="font-medium text-gray-900 mb-2">Analysis Parameters</h4>
              <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div>
                  <span class="text-gray-700">Risk Levels:</span>
                  <span class="font-medium">{{ results.suspect.parameters.risk_levels.join(', ') }}</span>
                </div>
                <div>
                  <span class="text-gray-700">Days Threshold:</span>
                  <span class="font-medium">{{ results.suspect.parameters.days_threshold }}</span>
                </div>
                <div>
                  <span class="text-gray-700">Include Arrests:</span>
                  <span class="font-medium">{{ results.suspect.parameters.include_arrests ? 'Yes' : 'No' }}</span>
                </div>
              </div>
            </div>

            <div class="mt-4 text-xs text-gray-500">
              Generated: {{ formatDate(results.suspect.generated_at) }}
            </div>
          </div>
        </div>

        <!-- Error Messages -->
        <div v-if="errorMessage" class="bg-red-50 border border-red-200 rounded-md p-4 mb-6">
          <div class="flex">
            <div class="flex-shrink-0">
              <svg class="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" />
              </svg>
            </div>
            <div class="ml-3">
              <p class="text-sm font-medium text-red-800">{{ errorMessage }}</p>
            </div>
          </div>
        </div>

        <!-- Quick Actions -->
        <div v-if="results.theft || results.suspect" class="card">
          <h3 class="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
          <div class="flex flex-wrap gap-4">
            <NuxtLink to="/maps" class="btn-primary">
              🗺️ Generate Maps
            </NuxtLink>
            <button @click="exportResults" class="btn-secondary">
              📊 Export Results
            </button>
            <button @click="clearResults" class="btn-secondary">
              🗑️ Clear Results
            </button>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script setup>
// Page meta
definePageMeta({
  title: 'Analysis - Auto Thievia'
})

// Composables
const config = useRuntimeConfig()
const apiBase = config.public.apiBase

// Reactive data
const analysisLoading = ref({
  theft: false,
  suspect: false
})

const theftAnalysis = ref({
  center_lat: 40.7357,
  center_lon: -74.1723,
  radius_km: 10.0
})

const suspectAnalysis = ref({
  risk_levels: ['High', 'Critical'],
  days_threshold: 30,
  include_arrests: true
})

const riskLevels = ['Low', 'Medium', 'High', 'Critical']

const results = ref({
  theft: null,
  suspect: null
})

const errorMessage = ref('')

// Methods
const clearError = () => {
  errorMessage.value = ''
}

const showError = (message) => {
  errorMessage.value = message
  setTimeout(() => errorMessage.value = '', 5000)
}

const runTheftAnalysis = async () => {
  try {
    analysisLoading.value.theft = true
    clearError()
    
    const response = await $fetch(`${apiBase}/analysis/theft`, {
      method: 'GET',
      query: theftAnalysis.value
    })
    
    results.value.theft = response
    
  } catch (error) {
    console.error('Error running theft analysis:', error)
    showError('Failed to run theft analysis. Please try again.')
  } finally {
    analysisLoading.value.theft = false
  }
}

const runSuspectAnalysis = async () => {
  try {
    analysisLoading.value.suspect = true
    clearError()
    
    const response = await $fetch(`${apiBase}/analysis/suspects`, {
      method: 'GET',
      query: suspectAnalysis.value
    })
    
    results.value.suspect = response
    
  } catch (error) {
    console.error('Error running suspect analysis:', error)
    showError('Failed to run suspect analysis. Please try again.')
  } finally {
    analysisLoading.value.suspect = false
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

const exportResults = () => {
  const data = {
    theft_analysis: results.value.theft,
    suspect_analysis: results.value.suspect,
    exported_at: new Date().toISOString()
  }
  
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `auto_thievia_analysis_${new Date().toISOString().split('T')[0]}.json`
  a.click()
  URL.revokeObjectURL(url)
}

const clearResults = () => {
  results.value = {
    theft: null,
    suspect: null
  }
}
</script>
