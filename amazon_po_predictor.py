import React, { useState, useEffect, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { Upload, Settings, TrendingUp, Package, AlertTriangle, Download } from 'lucide-react';

const AmazonPOPredictor = () => {
  // State management
  const [csvData, setCsvData] = useState(null);
  const [asinData, setAsinData] = useState([]);
  const [selectedAsin, setSelectedAsin] = useState('');
  
  // Model parameters
  const [forecastModel, setForecastModel] = useState('moving_average');
  const [movingAvgPeriods, setMovingAvgPeriods] = useState(4);
  const [exponentialAlpha, setExponentialAlpha] = useState(0.3);
  const [trendAlpha, setTrendAlpha] = useState(0.2);
  const [seasonalityPeriod, setSeasonalityPeriod] = useState(12);
  
  // Business parameters
  const [currentInventory, setCurrentInventory] = useState(1000);
  const [asinPrice, setAsinPrice] = useState(25.99);
  const [targetWoh, setTargetWoh] = useState(4);
  const [leadTimeWeeks, setLeadTimeWeeks] = useState(3);
  const [forecastHorizon, setForecastHorizon] = useState(26);
  const [safetyStockWeeks, setSafetyStockWeeks] = useState(1);
  const [minOrderQuantity, setMinOrderQuantity] = useState(500);
  const [maxOrderQuantity, setMaxOrderQuantity] = useState(10000);
  
  // Results
  const [predictions, setPredictions] = useState([]);
  const [poRecommendations, setPoRecommendations] = useState([]);

  // Parse CSV data
  const parseCSVData = (csvText) => {
    const lines = csvText.split('\n');
    const headers = lines[1]?.split(';') || [];
    
    if (headers.length < 4) return [];
    
    const weekColumns = headers.slice(3).filter(col => col.startsWith('Week'));
    const dataLine = lines[2]?.split(';') || [];
    
    if (dataLine.length < 4) return [];
    
    const asin = dataLine[0];
    const productTitle = dataLine[1];
    const brand = dataLine[2];
    
    const weeklyData = weekColumns.map((weekHeader, index) => {
      const value = parseFloat(dataLine[3 + index]?.replace(',', '')) || 0;
      const weekMatch = weekHeader.match(/Week (\d+)/);
      const weekNumber = weekMatch ? parseInt(weekMatch[1]) : index;
      return { week: weekNumber, forecast: value, weekHeader };
    });
    
    return [{
      asin,
      productTitle,
      brand,
      weeklyData: weeklyData.filter(w => w.forecast > 0)
    }];
  };

  // Forecasting models
  const applyForecastModel = (historicalData, horizon) => {
    if (!historicalData || historicalData.length === 0) return [];
    
    const forecasts = [];
    const values = historicalData.map(d => d.forecast);
    
    switch (forecastModel) {
      case 'moving_average':
        for (let i = 0; i < horizon; i++) {
          const recentValues = values.slice(-movingAvgPeriods);
          const avg = recentValues.reduce((a, b) => a + b, 0) / recentValues.length;
          forecasts.push(avg);
          values.push(avg);
        }
        break;
        
      case 'exponential_smoothing':
        let lastForecast = values[values.length - 1];
        for (let i = 0; i < horizon; i++) {
          forecasts.push(lastForecast);
        }
        break;
        
      case 'linear_trend':
        const n = values.length;
        const sumX = (n * (n + 1)) / 2;
        const sumXX = (n * (n + 1) * (2 * n + 1)) / 6;
        const sumY = values.reduce((a, b) => a + b, 0);
        const sumXY = values.reduce((sum, val, idx) => sum + val * (idx + 1), 0);
        
        const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        const intercept = (sumY - slope * sumX) / n;
        
        for (let i = 0; i < horizon; i++) {
          const forecast = intercept + slope * (n + i + 1);
          forecasts.push(Math.max(0, forecast));
        }
        break;
        
      case 'seasonal_naive':
        const seasonLength = Math.min(seasonalityPeriod, values.length);
        for (let i = 0; i < horizon; i++) {
          const seasonalIndex = i % seasonLength;
          const seasonalValue = values[values.length - seasonLength + seasonalIndex] || values[values.length - 1];
          forecasts.push(seasonalValue);
        }
        break;
        
      default:
        const avgValue = values.reduce((a, b) => a + b, 0) / values.length;
        for (let i = 0; i < horizon; i++) {
          forecasts.push(avgValue);
        }
    }
    
    return forecasts;
  };

  // Calculate PO requirements
  const calculatePORequirements = (historicalData, forecastedValues) => {
    if (!forecastedValues || forecastedValues.length === 0) return [];
    
    const poData = [];
    let currentInv = currentInventory;
    
    for (let week = 0; week < forecastedValues.length; week++) {
      const weeklyDemand = forecastedValues[week];
      const projectedInventory = currentInv - weeklyDemand;
      
      // Calculate target inventory (WOH * average weekly demand)
      const avgWeeklyDemand = forecastedValues.slice(Math.max(0, week - 3), week + 1)
        .reduce((a, b) => a + b, 0) / Math.min(4, week + 1);
      
      const targetInventory = targetWoh * avgWeeklyDemand;
      const safetyStock = safetyStockWeeks * avgWeeklyDemand;
      const reorderPoint = (leadTimeWeeks * avgWeeklyDemand) + safetyStock;
      
      let poQuantity = 0;
      let poValue = 0;
      let orderTrigger = false;
      
      // Check if we need to place an order
      if (projectedInventory <= reorderPoint) {
        orderTrigger = true;
        poQuantity = Math.max(
          minOrderQuantity,
          Math.min(maxOrderQuantity, targetInventory - projectedInventory + (leadTimeWeeks * avgWeeklyDemand))
        );
        poValue = poQuantity * asinPrice;
        currentInv += poQuantity;
      }
      
      currentInv = Math.max(0, projectedInventory);
      
      poData.push({
        week: week + 1,
        demand: Math.round(weeklyDemand),
        inventory: Math.round(currentInv),
        targetInventory: Math.round(targetInventory),
        reorderPoint: Math.round(reorderPoint),
        poQuantity: Math.round(poQuantity),
        poValue: Math.round(poValue),
        orderTrigger,
        woh: currentInv / avgWeeklyDemand
      });
    }
    
    return poData;
  };

  // Process data when parameters change
  useEffect(() => {
    if (asinData.length > 0 && selectedAsin) {
      const asin = asinData.find(a => a.asin === selectedAsin);
      if (asin && asin.weeklyData) {
        const forecasts = applyForecastModel(asin.weeklyData, forecastHorizon);
        const poData = calculatePORequirements(asin.weeklyData, forecasts);
        
        setPredictions(forecasts.map((f, i) => ({
          week: i + 1,
          forecast: Math.round(f),
          type: 'forecast'
        })));
        
        setPoRecommendations(poData);
      }
    }
  }, [asinData, selectedAsin, forecastModel, movingAvgPeriods, exponentialAlpha, 
      currentInventory, targetWoh, leadTimeWeeks, forecastHorizon, safetyStockWeeks,
      minOrderQuantity, maxOrderQuantity, asinPrice, trendAlpha, seasonalityPeriod]);

  // File upload handler
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (file) {
      try {
        const text = await file.text();
        const parsed = parseCSVData(text);
        setCsvData(text);
        setAsinData(parsed);
        if (parsed.length > 0) {
          setSelectedAsin(parsed[0].asin);
        }
      } catch (error) {
        console.error('Error parsing CSV:', error);
      }
    }
  };

  // Calculate summary metrics
  const summaryMetrics = useMemo(() => {
    if (poRecommendations.length === 0) return {};
    
    const totalPOValue = poRecommendations.reduce((sum, po) => sum + po.poValue, 0);
    const totalPOQuantity = poRecommendations.reduce((sum, po) => sum + po.poQuantity, 0);
    const avgWOH = poRecommendations.reduce((sum, po) => sum + (po.woh || 0), 0) / poRecommendations.length;
    const stockoutRisk = poRecommendations.filter(po => po.inventory <= 0).length;
    
    return {
      totalPOValue: Math.round(totalPOValue),
      totalPOQuantity: Math.round(totalPOQuantity),
      avgWOH: Math.round(avgWOH * 10) / 10,
      stockoutRisk
    };
  }, [poRecommendations]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-4">
            Amazon PO Replenishment Predictor
          </h1>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto">
            Advanced forecasting and inventory optimization for Amazon FBA replenishment planning
          </p>
        </div>

        {/* File Upload */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
          <div className="flex items-center mb-4">
            <Upload className="text-blue-600 mr-3" size={24} />
            <h2 className="text-xl font-semibold text-gray-800">Upload Forecast Data</h2>
          </div>
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
            <input
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
              className="hidden"
              id="csvUpload"
            />
            <label
              htmlFor="csvUpload"
              className="cursor-pointer flex flex-col items-center"
            >
              <Upload className="text-gray-400 mb-4" size={48} />
              <span className="text-lg font-medium text-gray-600 mb-2">
                Upload your CSV forecast file
              </span>
              <span className="text-sm text-gray-500">
                CSV files with ASIN, Product Title, Brand, and weekly forecasts
              </span>
            </label>
          </div>
          {asinData.length > 0 && (
            <div className="mt-4 p-4 bg-green-50 rounded-lg">
              <p className="text-green-800">
                ✅ Successfully loaded {asinData.length} ASIN(s)
              </p>
            </div>
          )}
        </div>

        {/* Configuration Panel */}
        {asinData.length > 0 && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
            {/* ASIN Selection & Business Parameters */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <div className="flex items-center mb-4">
                <Package className="text-green-600 mr-3" size={24} />
                <h3 className="text-lg font-semibold text-gray-800">Product & Inventory</h3>
              </div>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Select ASIN
                  </label>
                  <select
                    value={selectedAsin}
                    onChange={(e) => setSelectedAsin(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    {asinData.map(asin => (
                      <option key={asin.asin} value={asin.asin}>
                        {asin.asin} - {asin.brand}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Current Inventory
                  </label>
                  <input
                    type="number"
                    value={currentInventory}
                    onChange={(e) => setCurrentInventory(Number(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    ASIN Price ($)
                  </label>
                  <input
                    type="number"
                    step="0.01"
                    value={asinPrice}
                    onChange={(e) => setAsinPrice(Number(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Target WOH (Weeks on Hand)
                  </label>
                  <input
                    type="number"
                    step="0.5"
                    value={targetWoh}
                    onChange={(e) => setTargetWoh(Number(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Lead Time (Weeks)
                  </label>
                  <input
                    type="number"
                    value={leadTimeWeeks}
                    onChange={(e) => setLeadTimeWeeks(Number(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              </div>
            </div>

            {/* Forecast Model Parameters */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <div className="flex items-center mb-4">
                <TrendingUp className="text-purple-600 mr-3" size={24} />
                <h3 className="text-lg font-semibold text-gray-800">Forecast Model</h3>
              </div>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Forecasting Method
                  </label>
                  <select
                    value={forecastModel}
                    onChange={(e) => setForecastModel(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="moving_average">Moving Average</option>
                    <option value="exponential_smoothing">Exponential Smoothing</option>
                    <option value="linear_trend">Linear Trend</option>
                    <option value="seasonal_naive">Seasonal Naive</option>
                  </select>
                </div>

                {forecastModel === 'moving_average' && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Moving Average Periods
                    </label>
                    <input
                      type="number"
                      min="1"
                      max="12"
                      value={movingAvgPeriods}
                      onChange={(e) => setMovingAvgPeriods(Number(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                )}

                {forecastModel === 'exponential_smoothing' && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Alpha (0.1-0.9)
                    </label>
                    <input
                      type="number"
                      min="0.1"
                      max="0.9"
                      step="0.1"
                      value={exponentialAlpha}
                      onChange={(e) => setExponentialAlpha(Number(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                )}

                {forecastModel === 'seasonal_naive' && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Seasonality Period
                    </label>
                    <input
                      type="number"
                      min="4"
                      max="52"
                      value={seasonalityPeriod}
                      onChange={(e) => setSeasonalityPeriod(Number(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                )}

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Forecast Horizon (Weeks)
                  </label>
                  <input
                    type="number"
                    min="4"
                    max="52"
                    value={forecastHorizon}
                    onChange={(e) => setForecastHorizon(Number(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              </div>
            </div>

            {/* Advanced Parameters */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <div className="flex items-center mb-4">
                <Settings className="text-orange-600 mr-3" size={24} />
                <h3 className="text-lg font-semibold text-gray-800">Advanced Settings</h3>
              </div>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Safety Stock (Weeks)
                  </label>
                  <input
                    type="number"
                    step="0.5"
                    value={safetyStockWeeks}
                    onChange={(e) => setSafetyStockWeeks(Number(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Min Order Quantity
                  </label>
                  <input
                    type="number"
                    value={minOrderQuantity}
                    onChange={(e) => setMinOrderQuantity(Number(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Max Order Quantity
                  </label>
                  <input
                    type="number"
                    value={maxOrderQuantity}
                    onChange={(e) => setMaxOrderQuantity(Number(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Summary Metrics */}
        {poRecommendations.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div className="bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg shadow-lg p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-blue-100 text-sm font-medium">Total PO Value</p>
                  <p className="text-2xl font-bold">${summaryMetrics.totalPOValue?.toLocaleString()}</p>
                </div>
                <Package className="text-blue-200" size={32} />
              </div>
            </div>

            <div className="bg-gradient-to-r from-green-500 to-green-600 text-white rounded-lg shadow-lg p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-green-100 text-sm font-medium">Total PO Quantity</p>
                  <p className="text-2xl font-bold">{summaryMetrics.totalPOQuantity?.toLocaleString()}</p>
                </div>
                <TrendingUp className="text-green-200" size={32} />
              </div>
            </div>

            <div className="bg-gradient-to-r from-purple-500 to-purple-600 text-white rounded-lg shadow-lg p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-purple-100 text-sm font-medium">Avg WOH</p>
                  <p className="text-2xl font-bold">{summaryMetrics.avgWOH}</p>
                </div>
                <Settings className="text-purple-200" size={32} />
              </div>
            </div>

            <div className="bg-gradient-to-r from-red-500 to-red-600 text-white rounded-lg shadow-lg p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-red-100 text-sm font-medium">Stockout Risk</p>
                  <p className="text-2xl font-bold">{summaryMetrics.stockoutRisk} weeks</p>
                </div>
                <AlertTriangle className="text-red-200" size={32} />
              </div>
            </div>
          </div>
        )}

        {/* Charts */}
        {poRecommendations.length > 0 && (
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-8 mb-8">
            {/* Demand Forecast Chart */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-xl font-semibold text-gray-800 mb-4">
                Demand Forecast & Inventory Levels
              </h3>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={poRecommendations}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="week" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="demand" 
                    stroke="#3B82F6" 
                    strokeWidth={3}
                    name="Weekly Demand"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="inventory" 
                    stroke="#10B981" 
                    strokeWidth={2}
                    name="Inventory Level"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="reorderPoint" 
                    stroke="#F59E0B" 
                    strokeDasharray="5 5"
                    name="Reorder Point"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* PO Quantities Chart */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-xl font-semibold text-gray-800 mb-4">
                Purchase Order Quantities
              </h3>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={poRecommendations.filter(po => po.poQuantity > 0)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="week" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar 
                    dataKey="poQuantity" 
                    fill="#8B5CF6" 
                    name="PO Quantity"
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* PO Schedule Table */}
        {poRecommendations.length > 0 && (
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-semibold text-gray-800">
                Purchase Order Schedule
              </h3>
              <button className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                <Download className="mr-2" size={16} />
                Export CSV
              </button>
            </div>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Week
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Demand
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Inventory
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      WOH
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      PO Qty
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      PO Value
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Status
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {poRecommendations.slice(0, 12).map((po, index) => (
                    <tr key={index} className={po.orderTrigger ? 'bg-yellow-50' : ''}>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        Week {po.week}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {po.demand.toLocaleString()}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {po.inventory.toLocaleString()}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {po.woh?.toFixed(1) || '0.0'}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {po.poQuantity > 0 ? po.poQuantity.toLocaleString() : '-'}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {po.poValue > 0 ? `${po.poValue.toLocaleString()}` : '-'}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        {po.orderTrigger ? (
                          <span className="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-yellow-100 text-yellow-800">
                            Order Required
                          </span>
                        ) : po.inventory <= po.reorderPoint ? (
                          <span className="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-red-100 text-red-800">
                            Low Stock
                          </span>
                        ) : (
                          <span className="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-green-100 text-green-800">
                            Normal
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {poRecommendations.length > 12 && (
              <div className="mt-4 text-center text-sm text-gray-500">
                Showing first 12 weeks. Export CSV for complete data.
              </div>
            )}
          </div>
        )}

        {/* Instructions */}
        {asinData.length === 0 && (
          <div className="bg-white rounded-lg shadow-lg p-8">
            <h3 className="text-2xl font-semibold text-gray-800 mb-6">How to Use</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="text-center">
                <div className="bg-blue-100 rounded-full p-4 w-16 h-16 mx-auto mb-4 flex items-center justify-center">
                  <Upload className="text-blue-600" size={32} />
                </div>
                <h4 className="font-semibold text-gray-800 mb-2">1. Upload Data</h4>
                <p className="text-sm text-gray-600">
                  Upload your CSV file with ASIN forecast data including weekly sellout projections.
                </p>
              </div>
              
              <div className="text-center">
                <div className="bg-green-100 rounded-full p-4 w-16 h-16 mx-auto mb-4 flex items-center justify-center">
                  <Settings className="text-green-600" size={32} />
                </div>
                <h4 className="font-semibold text-gray-800 mb-2">2. Configure Parameters</h4>
                <p className="text-sm text-gray-600">
                  Set your inventory levels, pricing, lead times, and target weeks on hand.
                </p>
              </div>
              
              <div className="text-center">
                <div className="bg-purple-100 rounded-full p-4 w-16 h-16 mx-auto mb-4 flex items-center justify-center">
                  <TrendingUp className="text-purple-600" size={32} />
                </div>
                <h4 className="font-semibold text-gray-800 mb-2">3. Choose Model</h4>
                <p className="text-sm text-gray-600">
                  Select from multiple forecasting models and tune parameters for optimal accuracy.
                </p>
              </div>
              
              <div className="text-center">
                <div className="bg-orange-100 rounded-full p-4 w-16 h-16 mx-auto mb-4 flex items-center justify-center">
                  <Package className="text-orange-600" size={32} />
                </div>
                <h4 className="font-semibold text-gray-800 mb-2">4. Get Recommendations</h4>
                <p className="text-sm text-gray-600">
                  View detailed PO recommendations, timing, quantities, and inventory projections.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Features */}
        <div className="bg-white rounded-lg shadow-lg p-8 mt-8">
          <h3 className="text-2xl font-semibold text-gray-800 mb-6">Key Features</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div className="flex items-start">
              <TrendingUp className="text-blue-600 mr-3 mt-1 flex-shrink-0" size={20} />
              <div>
                <h4 className="font-semibold text-gray-800 mb-1">Multiple Forecast Models</h4>
                <p className="text-sm text-gray-600">
                  Choose from Moving Average, Exponential Smoothing, Linear Trend, or Seasonal Naive models.
                </p>
              </div>
            </div>
            
            <div className="flex items-start">
              <Package className="text-green-600 mr-3 mt-1 flex-shrink-0" size={20} />
              <div>
                <h4 className="font-semibold text-gray-800 mb-1">Inventory Optimization</h4>
                <p className="text-sm text-gray-600">
                  Automatic calculation of reorder points, safety stock, and target inventory levels.
                </p>
              </div>
            </div>
            
            <div className="flex items-start">
              <AlertTriangle className="text-orange-600 mr-3 mt-1 flex-shrink-0" size={20} />
              <div>
                <h4 className="font-semibold text-gray-800 mb-1">Risk Management</h4>
                <p className="text-sm text-gray-600">
                  Identifies stockout risks and provides early warning alerts for inventory management.
                </p>
              </div>
            </div>
            
            <div className="flex items-start">
              <Settings className="text-purple-600 mr-3 mt-1 flex-shrink-0" size={20} />
              <div>
                <h4 className="font-semibold text-gray-800 mb-1">Flexible Parameters</h4>
                <p className="text-sm text-gray-600">
                  Customize lead times, order quantities, pricing, and weeks on hand targets.
                </p>
              </div>
            </div>
            
            <div className="flex items-start">
              <Download className="text-indigo-600 mr-3 mt-1 flex-shrink-0" size={20} />
              <div>
                <h4 className="font-semibold text-gray-800 mb-1">Export Capabilities</h4>
                <p className="text-sm text-gray-600">
                  Download detailed PO schedules and recommendations for operational planning.
                </p>
              </div>
            </div>
            
            <div className="flex items-start">
              <div className="w-5 h-5 bg-gradient-to-r from-blue-500 to-purple-600 rounded mr-3 mt-1 flex-shrink-0"></div>
              <div>
                <h4 className="font-semibold text-gray-800 mb-1">Interactive Visualizations</h4>
                <p className="text-sm text-gray-600">
                  Real-time charts and graphs to visualize demand patterns and inventory flows.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-12 pb-8">
          <p className="text-gray-600">
            Built for Amazon FBA sellers and inventory managers. Deploy easily on Streamlit or any React hosting platform.
          </p>
          <div className="mt-4 flex justify-center space-x-4 text-sm text-gray-500">
            <span>• Real-time forecasting</span>
            <span>• Inventory optimization</span>
            <span>• PO automation</span>
            <span>• Risk management</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AmazonPOPredictor;
