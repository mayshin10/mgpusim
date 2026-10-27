package fabric

import (
	"github.com/sarchlab/akita/v5/messaging"
	"github.com/sarchlab/akita/v5/modeling"
	"github.com/sarchlab/akita/v5/timing"
)

var defaultSpec = Spec{
	Freq:               1 * timing.GHz,
	HeaderBytes:        16,
	MaxInflightPerPort: 256,
}

// DefaultSpec returns a copy of the default configuration.
func DefaultSpec() Spec {
	return defaultSpec
}

// Builder builds fabrics.
type Builder struct {
	spec      Spec
	registrar modeling.Registrar
}

// MakeBuilder creates a builder with the default configuration.
func MakeBuilder() Builder {
	return Builder{spec: defaultSpec}
}

// WithRegistrar wires the builder to a registrar.
func (b Builder) WithRegistrar(reg modeling.Registrar) Builder {
	b.registrar = reg
	return b
}

// WithSpec sets the entire configuration. Start from DefaultSpec() and tweak.
func (b Builder) WithSpec(spec Spec) Builder {
	b.spec = spec
	return b
}

// Build builds the fabric. Nodes and links are added afterwards with AddNode
// and AddLink, and ports are attached with PlugInAt.
func (b Builder) Build(name string) *Comp {
	if b.registrar == nil {
		panic("fabric: WithRegistrar is required")
	}

	engine := b.registrar.GetEngine()
	spec := b.spec

	modelComp := modeling.NewBuilder[Spec, State, modeling.None]().
		WithEngine(engine).
		WithFreq(spec.Freq).
		WithSpec(spec).
		Build(name)

	// Like directconnection, the fabric ticks after the primary components.
	modelComp.TickingComponent = modeling.NewSecondaryTickingComponent(
		name, engine, spec.Freq, modelComp)

	mw := &middleware{
		comp:      modelComp,
		engine:    engine,
		portIndex: make(map[messaging.RemotePort]int),
	}
	modelComp.AddMiddleware(mw)

	conn := &Comp{Component: modelComp}

	b.registrar.RegisterConnection(conn)

	return conn
}
