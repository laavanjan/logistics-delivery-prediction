# =============================================================================
# Test Suite for XGBoost Delivery Time Prediction API
# Team: Laavanjan & Vidushi
# =============================================================================

import asyncio
import json
import requests
import time
from typing import Dict, Any

# Test Configuration
API_BASE_URL = "http://localhost:8000"


class APITester:
    """Test class for API endpoints"""

    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()

    def test_health_endpoint(self) -> bool:
        """Test the health check endpoint"""
        print("🏥 Testing health endpoint...")

        try:
            response = self.session.get(f"{self.base_url}/health")

            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data['status']}")
                print(f"   Model loaded: {data['model_loaded']}")
                print(f"   Uptime: {data['uptime_seconds']:.2f}s")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False

    def test_single_prediction(self) -> bool:
        """Test single prediction endpoint"""
        print("🎯 Testing single prediction endpoint...")

        # Sample test data
        test_data = {
            "osrm_distance": 5000.0,
            "osrm_time": 900.0,
            "actual_distance_to_destination": 4800.0,
            "cutoff_factor": 1.2,
            "factor": 0.8,
            "time_difference": 120.0,
            "is_cutoff": False,
            "is_heavy_delay": False,
            "destination_center": "Delhi_Hub",
            "destination_name": "Customer_Location_1",
        }

        try:
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/predict",
                json=test_data,
                headers={"Content-Type": "application/json"},
            )
            response_time = (time.time() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                prediction = data["prediction"]

                print(f"✅ Single prediction successful!")
                print(f"   Predicted time: {prediction['predicted_time_formatted']}")
                print(f"   Seconds: {prediction['predicted_time_seconds']:.2f}")
                print(f"   Minutes: {prediction['predicted_time_minutes']:.2f}")
                print(f"   Response time: {response_time:.2f}ms")
                print(f"   Request ID: {data['request_id']}")
                return True
            else:
                print(f"❌ Single prediction failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False

        except Exception as e:
            print(f"❌ Single prediction error: {e}")
            return False

    def test_batch_prediction(self) -> bool:
        """Test batch prediction endpoint"""
        print("📦 Testing batch prediction endpoint...")

        # Sample batch test data
        batch_data = {
            "deliveries": [
                {
                    "osrm_distance": 3000.0,
                    "osrm_time": 600.0,
                    "actual_distance_to_destination": 2900.0,
                    "cutoff_factor": 1.0,
                    "factor": 0.9,
                },
                {
                    "osrm_distance": 7000.0,
                    "osrm_time": 1200.0,
                    "actual_distance_to_destination": 6800.0,
                    "cutoff_factor": 1.5,
                    "factor": 1.1,
                    "is_cutoff": True,
                },
                {
                    "osrm_distance": 2000.0,
                    "osrm_time": 400.0,
                    "actual_distance_to_destination": 1950.0,
                    "cutoff_factor": 0.8,
                    "factor": 0.7,
                },
            ]
        }

        try:
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/predict/batch",
                json=batch_data,
                headers={"Content-Type": "application/json"},
            )
            response_time = (time.time() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                predictions = data["predictions"]

                print(f"✅ Batch prediction successful!")
                print(f"   Total processed: {data['total_processed']}")
                print(f"   Response time: {response_time:.2f}ms")
                print(
                    f"   Avg time per prediction: {response_time/len(predictions):.2f}ms"
                )

                for i, pred in enumerate(predictions, 1):
                    print(f"   Delivery {i}: {pred['predicted_time_formatted']}")

                return True
            else:
                print(f"❌ Batch prediction failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False

        except Exception as e:
            print(f"❌ Batch prediction error: {e}")
            return False

    def test_model_info(self) -> bool:
        """Test model info endpoint"""
        print("ℹ️ Testing model info endpoint...")

        try:
            response = self.session.get(f"{self.base_url}/model/info")

            if response.status_code == 200:
                data = response.json()
                print(f"✅ Model info retrieved successfully!")
                print(f"   Model type: {data['model_type']}")
                print(f"   Version: {data['version']}")
                print(f"   Features count: {data['features_count']}")
                print(f"   Model loaded: {data['model_loaded']}")
                return True
            else:
                print(f"❌ Model info failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Model info error: {e}")
            return False

    def test_error_handling(self) -> bool:
        """Test error handling with invalid data"""
        print("⚠️ Testing error handling...")

        # Test with missing required fields
        invalid_data = {
            "osrm_distance": 5000.0,
            # Missing osrm_time and actual_distance_to_destination
        }

        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=invalid_data,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 422:  # Validation error
                print(f"✅ Error handling works correctly!")
                print(f"   Status code: {response.status_code}")
                return True
            else:
                print(
                    f"❌ Unexpected response for invalid data: {response.status_code}"
                )
                return False

        except Exception as e:
            print(f"❌ Error handling test error: {e}")
            return False

    def run_all_tests(self) -> Dict[str, bool]:
        """Run all API tests"""
        print("🚀 Starting API Test Suite...")
        print("=" * 60)

        results = {}

        # Test each endpoint
        results["health"] = self.test_health_endpoint()
        print()

        results["single_prediction"] = self.test_single_prediction()
        print()

        results["batch_prediction"] = self.test_batch_prediction()
        print()

        results["model_info"] = self.test_model_info()
        print()

        results["error_handling"] = self.test_error_handling()
        print()

        # Summary
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)

        passed = sum(results.values())
        total = len(results)

        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name:<20}: {status}")

        print(f"\nOverall: {passed}/{total} tests passed")

        if passed == total:
            print("🎉 All tests passed! API is working correctly.")
        else:
            print("⚠️ Some tests failed. Please check the API implementation.")

        return results


def main():
    """Main function to run tests"""
    print("🧪 XGBoost Delivery Time Prediction API - Test Suite")
    print("=" * 60)
    print("Make sure the API server is running on http://localhost:8000")
    print("Start the server with: python api_app.py")
    print()

    # Wait for user confirmation
    input("Press Enter to start testing...")
    print()

    # Run tests
    tester = APITester()
    results = tester.run_all_tests()

    return results


if __name__ == "__main__":
    main()
