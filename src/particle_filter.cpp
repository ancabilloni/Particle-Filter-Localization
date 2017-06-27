/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 200;
	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for(int i=0; i<num_particles; ++i){
		Particle particle_temp;
		particle_temp.id = i;
		particle_temp.x = dist_x(gen);
		particle_temp.y = dist_y(gen);
		particle_temp.theta = dist_theta(gen);
		particle_temp.weight = 1;
		particles.push_back(particle_temp);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;

	for(int i=0; i<num_particles; ++i){
		// Gaussian Noises

		normal_distribution<double> noise_x(0, std_pos[0]);
		normal_distribution<double> noise_y(0, std_pos[1]);
		normal_distribution<double> noise_theta(0, std_pos[2]);

		// Predict new particle's position base on velocity and yaw rate
		if(fabs(yaw_rate) > 0.01){
			particles[i].x += (velocity/yaw_rate)*(sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
			particles[i].y += (velocity/yaw_rate)*(cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));

		}
		else{
			particles[i].x += velocity*delta_t*cos(particles[i].theta);
			particles[i].y += velocity*delta_t*sin(particles[i].theta);
		}
		particles[i].theta += yaw_rate*delta_t + noise_theta(gen);
		particles[i].x += noise_x(gen);
		particles[i].y += noise_y(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	weights.clear();
	for(int i=0; i<num_particles; ++i){
		// Current particle filter values
		double particle_x = particles[i].x;
		double particle_y = particles[i].y;
		double  particle_theta = particles[i].theta;

		// Predicted particle filter to observed landmarks & Landmarks in range

		vector<LandmarkObs> LandmarksInRange;
		// List of landmarks within sensor range
		for(int lm=0; lm<map_landmarks.landmark_list.size(); ++lm){
			double distance = dist(particle_x, particle_y, map_landmarks.landmark_list[lm].x_f,
								   map_landmarks.landmark_list[lm].y_f);

			if(distance <= sensor_range){
				LandmarkObs temp_landmark;
				temp_landmark.x = map_landmarks.landmark_list[lm].x_f;
				temp_landmark.y = map_landmarks.landmark_list[lm].y_f;
				temp_landmark.id = map_landmarks.landmark_list[lm].id_i;
				LandmarksInRange.push_back(temp_landmark);
			}
		}

		// Predict particle measurements and nearest neighbor
		vector<LandmarkObs> predicted;
		vector<LandmarkObs> nearest_neighbor;
		for(int j=0; j<observations.size(); ++j) {
			double predict_x, predict_y;
			int id = 0;
			// Predict particle measurements in global coordinate
			predict_x = observations[j].x * cos(particle_theta) - observations[j].y * sin(particle_theta) + particle_x;
			predict_y = observations[j].x * sin(particle_theta) + observations[j].y * cos(particle_theta) + particle_y;

			// Find the nearest landmark
			double min_dist = INFINITY;
			double mean_x, mean_y;
			mean_x = 0.0;
			mean_y = 0.0;
			for (int m = 0; m < LandmarksInRange.size(); ++m) {
				double dist_landmark = dist(predict_x, predict_y, LandmarksInRange[m].x, LandmarksInRange[m].y);
				if (dist_landmark < min_dist) {
					mean_x = LandmarksInRange[m].x;
					mean_y = LandmarksInRange[m].y;
					id = LandmarksInRange[m].id;
					min_dist = dist_landmark;
				}
			}

			LandmarkObs predict_temp;
			predict_temp.x = predict_x;
			predict_temp.y = predict_y;
			predict_temp.id = id;
			predicted.push_back(predict_temp);

			LandmarkObs nearest_temp;
			nearest_temp.x = mean_x;
			nearest_temp.y = mean_y;
			nearest_neighbor.push_back(nearest_temp);
		}

		// Update Weight
		double update_weight = 1.0;
		double std_x = std_landmark[0];
		double std_y = std_landmark[1];
		double normalizer = 1/(2.0*M_PI*std_x*std_y);

		for(int w = 0; w<predicted.size(); ++w){
			double dx = predicted[w].x - nearest_neighbor[w].x;
			double dy = predicted[w].y - nearest_neighbor[w].y;
			update_weight *= normalizer*exp(-(dx*dx/(2*std_x*std_x)) - (dy*dy/(2*std_y*std_y)));
		}
		particles[i].weight = update_weight;
		weights.push_back(particles[i].weight);
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<Particle> resampled_particles;
	default_random_engine gen;
	discrete_distribution<int> weights_distribution(weights.begin(),weights.end());
	for(int i=0; i<num_particles; ++i){
		resampled_particles.push_back(particles[weights_distribution(gen)]);
	}
	particles = resampled_particles;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
