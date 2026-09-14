/**
 * @brief User Interface for PUMI-Tally
 */

#ifndef PUMITALLY_PUMITALLY_H
#define PUMITALLY_PUMITALLY_H

#include <cstdint>
#include <memory>

/**
 * @brief PUMI-Tally Namespace
 * @details Provides API to PUMI-PiC for Parallel Unstructured Mesh
 * Tally Operations
 */
namespace pumitally {
/**
 * @brief Provides the functor to execute when particle reaches element boundary
 * @details
 * Evaluates flux and its auxiliary operations after crossing element
 * boundaries.
 */
struct ParticleAtElemBoundary;
struct PumiTallyImpl;

/**
 * @brief PUMI-Tally Interface Class
 * @details Provides user interface to call directly from physics application.
 * All functions take C++ builtin variable types for robustness. All PUMI-PiC
 * and Omega_h mesh related data are held inside the @ref PUMITallyImpl class.
 *
 * @see PUMITallyImpl
 */
class PumiTally {
public:
  /**
   * @brief Read mesh and initialize particles
   * @details Calls the @ref pimpl constructor to allocate particle DS and other
   * auxiliary arrays
   * @param mesh_filename Omega_h mesh filename (*.osh file)
   * @param num_particles Actual number of particles
   * @param argc Program argc
   * @param argv Program argv
   *
   * @note The mesh file name should end like `*.osh`, not `*.osh/`.
   * Autocomplete may sometimes add a `/` with the mesh name since Omega_h
   * meshes are stored as directories.
   * @n `argc` and `argv` are takes for MPI and Kokkos inputs.
   */
  PumiTally(const std::string &mesh_filename, int32_t num_particles, int &argc,
            char **&argv);

  /**
   * Perform the is_initial_track search
   * @param init_particle_positions Positions of the particles flattened as x1,
   * y1, z1, x2, y2, ...
   * @param size Number of particles
   *
   * @details
   * Monte Carlo physics codes generally samples the origin points based on the
   * users' choice and physics is involved in it. PUMI-Tally always needs to
   * know which mesh element they are currently in. Therefore, an initial search
   * is done to find the starting position of the particles. Nothing is tallied
   * during this search.
   *
   * @note Call this once for every new set of source particles, i.e. at the
   * start of every batch (and of every in flight sub iteration of a batch),
   * not only once at the start of the simulation. Otherwise the particles of
   * the following batches are tracked from wherever the previous batch left
   * them and the track lengths of their first step are tallied into the wrong
   * elements.
   */
  void CopyInitialPosition(double *init_particle_positions,
                           std::int32_t size) const;

  /**
   * Track particles to a new location and tally
   *
   * @param particle_origin Current particle positions flattened as x1, y1, z1,
   * x2, y2, ...
   * @param particle_destinations Particle destinations flattened origin
   * locations
   * @param flying If the particles are flying in this step 1-flying 0-stopped
   * @param weights Weight of particle (multiplied when tallying), usually [0,1]
   * @param size Number of particles
   *
   * @details
   * It first moves the particles to the origin locations. This step is
   * necessary because sometimes particles get reincarnated (for example in
   * OpenMC, particles get absorbed and resampled to a new location) and show up
   * at a new location. Then they move to the destination. They are not tallied
   * when moving to the origin (first step).
   */
  void MoveToNextLocation(double *particle_origin,
                          double *particle_destinations, int8_t *flying,
                          double *weights, int32_t size) const;

  /**
   * @brief Close the current batch and fold it into the running statistics
   *
   * @param normalization_factor Factor the raw batch tally is scaled with
   * before it is accumulated, for example
   * `total_source / (n_particles * gen_per_batch)` in OpenMC. Pass 1.0 if the
   * physics code does not normalize its tallies.
   *
   * @details
   * PUMI-Tally keeps a per-batch flux accumulator that holds
   * `sum(track_length * weight)` of the current batch only. This call scales it
   * by @p normalization_factor, adds the scaled value to the sum of the fluxes
   * and its square to the sum of the squared fluxes, and then zeroes the
   * per-batch accumulator. Those two sums are what lets @ref WriteTallyResults
   * report a mean and a standard deviation.
   *
   * @note Call it once per batch that should contribute to the result, after
   * all the particles of that batch are done moving.
   *
   * @see DiscardBatchTally
   */
  void AccumulateBatchTally(double normalization_factor) const;

  /**
   * @brief Drop the flux tallied in the current batch
   * @details
   * For batches that must not contribute to the reported result, such as the
   * inactive batches of an eigenvalue calculation. It only clears the
   * per-batch accumulator; the statistics collected from the previous batches
   * are kept.
   *
   * @see AccumulateBatchTally
   */
  void DiscardBatchTally() const;

  /**
   * @brief Write the mesh tally to VTU and Omega_h (.osh) file
   * @details
   * Writes the `flux` (mean over the accumulated batches) and `flux_std_dev`
   * (standard deviation of that mean) tags, both normalized by element volume,
   * along with the `volume` tag.
   *
   * @note If no batch was ever closed with @ref AccumulateBatchTally, whatever
   * was tallied so far is reported as a single unnormalized batch, which gives
   * the mean the previous single batch behaviour and a zero standard
   * deviation.
   */
  void WriteTallyResults() const;

  /**
   * @brief PUMI-Tally Destructor
   * @details Call at the end of tally operations to avoid memory leaks.
   *
   * @see Pumitally
   */
  ~PumiTally();

private:
  std::unique_ptr<PumiTallyImpl> pimpl_; //!< @ref PumiTallyImpl holder
};
} // namespace pumitally

#endif // PUMITALLY_PUMITALLY_H
